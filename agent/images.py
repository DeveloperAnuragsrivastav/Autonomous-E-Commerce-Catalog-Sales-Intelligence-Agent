"""Lifestyle images, one per SKU, saved as images/<SKU>.jpg.

Providers (tried in order with --image-provider auto):
  gemini     image-to-image edit (gemini-*-image): the vendor photo + a scene prompt. Photorealistic,
             but needs a billing-enabled key (free-tier quota for image models is 0).
  composite  fully local and free: the product is cut out of the vendor photo (rembg) and placed in a
             room scene rendered in perspective from the product's real dimensions, with contact and
             cast shadows. Product pixels are untouched, so appearance matches the source exactly.
When the gemini provider reports no quota, the run switches to composite for the remaining SKUs.
"""
from __future__ import annotations

import io
import logging
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFilter
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .config import Settings
from .deadline import call_with_deadline
from .models import Product

log = logging.getLogger(__name__)

W, H = 1600, 1200
IMAGE_TIMEOUT_S = 60
JPEG_QUALITY = 90


class ImageResult(BaseModel):
    sku: str
    status: str  # ok | skipped | error
    provider: str = ""
    path: str | None = None
    source_url: str | None = None
    note: str = ""


class ProviderUnavailable(Exception):
    """The provider cannot serve any request this run (no key, zero quota)."""


# --- orchestration --------------------------------------------------------------------------


class ImageGenerator:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.out_dir = settings.images_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._gemini_available = settings.image_provider in ("auto", "gemini") and bool(settings.google_api_key)
        self._gemini_reason = "" if settings.google_api_key else "GOOGLE_API_KEY not set"
        self._rembg = None
        self._lock = threading.Lock()

    def prefetch(self, products: list[Product]) -> None:
        """Download all source photos in parallel; failures are retried per SKU in generate()."""
        urls = [p.image_url for p in products if p.image_url]

        def safe(url: str) -> None:
            try:
                fetch_source(url, self.settings.cache_dir)
            except Exception as exc:
                log.warning("Prefetch failed for %s: %s", url, exc)

        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(safe, urls))

    def generate(self, product: Product) -> ImageResult:
        path = self.out_dir / f"{product.sku}.jpg"
        result = ImageResult(sku=product.sku, status="error", source_url=product.image_url)
        if not product.image_url:
            result.status, result.note = "skipped", "No source image URL in vendor sheet"
            return result
        if self.settings.use_cache and path.exists():
            meta = path.with_suffix(".json")
            cached = ImageResult.model_validate_json(meta.read_text()) if meta.exists() else None
            if cached and cached.status == "ok":
                cached.note = (cached.note + " (reused existing file)").strip()
                return cached
        try:
            source = load_image(fetch_source(product.image_url, self.settings.cache_dir))
        except Exception as exc:
            result.note = f"Could not download source image: {type(exc).__name__}: {exc}"
            log.warning("Source image download failed for %s: %s", product.sku, exc)
            return result

        image, provider, note = None, "", ""
        if self.settings.image_provider in ("auto", "gemini") and self._gemini_available:
            try:
                image, provider = self._gemini(product, source), f"gemini:{self.settings.image_model}"
            except ProviderUnavailable as exc:
                self._gemini_available, self._gemini_reason = False, str(exc)
                log.warning("Gemini image generation unavailable (%s); using composite provider", exc)
            except (httpx.TransportError, httpx.TimeoutException, TimeoutError, ConnectionError) as exc:
                # Unreachable endpoint (e.g. a stalled network path): pay the timeout once, not per SKU.
                self._gemini_available = False
                self._gemini_reason = f"Gemini image endpoint unreachable ({type(exc).__name__})"
                note = f"{self._gemini_reason}; used composite fallback"
                log.warning("%s; using composite provider for the rest of the run", self._gemini_reason)
            except Exception as exc:
                note = f"Gemini failed ({type(exc).__name__}); used composite fallback"
                log.warning("Gemini image failed for %s: %s", product.sku, exc)
        if image is None and self.settings.image_provider in ("auto", "composite"):
            if self._gemini_reason and not note:
                note = f"Composite provider (Gemini unavailable: {self._gemini_reason})"
            image, provider = self._composite(product, source), "composite:rembg+rendered-room"
        if image is None:
            result.note = note or f"Provider '{self.settings.image_provider}' unavailable: {self._gemini_reason}"
            return result

        image.convert("RGB").save(path, "JPEG", quality=JPEG_QUALITY, optimize=True)
        result = ImageResult(sku=product.sku, status="ok", provider=provider, path=str(path.relative_to(self.settings.images_dir.parent)),
                             source_url=product.image_url, note=note)
        path.with_suffix(".json").write_text(result.model_dump_json(indent=1))
        return result

    # --- gemini -------------------------------------------------------------------------

    def _gemini(self, product: Product, source: Image.Image) -> Image.Image:
        from google import genai
        from google.genai import errors, types

        client = genai.Client(api_key=self.settings.google_api_key,
                              http_options=types.HttpOptions(timeout=IMAGE_TIMEOUT_S * 1000))
        ref = source.copy()
        ref.thumbnail((1024, 1024))
        prompt = gemini_prompt(product)
        try:
            resp = _gemini_retry(lambda: call_with_deadline(lambda: client.models.generate_content(
                model=self.settings.image_model, contents=[ref, prompt],
                config=types.GenerateContentConfig(response_modalities=["IMAGE", "TEXT"]),
            ), IMAGE_TIMEOUT_S + 30, "gemini image"))
        except errors.APIError as exc:
            if exc.code == 429 and "limit: 0" in str(exc):
                raise ProviderUnavailable("image model quota is 0 on this key (free tier); enable billing") from exc
            if exc.code in (401, 403, 404):
                raise ProviderUnavailable(f"{exc.code} from image model") from exc
            raise
        for part in resp.candidates[0].content.parts:
            if part.inline_data:
                return Image.open(io.BytesIO(part.inline_data.data)).convert("RGB")
        raise RuntimeError("Gemini returned no image")

    # --- composite ----------------------------------------------------------------------

    def _composite(self, product: Product, source: Image.Image) -> Image.Image:
        with self._lock:
            if self._rembg is None:
                from rembg import new_session

                self._rembg = new_session("isnet-general-use")
        from rembg import remove

        src = source.copy()
        src.thumbnail((1600, 1600))
        cutout = remove(src, session=self._rembg, post_process_mask=True)
        cutout = cutout.crop(cutout.getbbox())
        return render_room_scene(product, cutout)


def gemini_prompt(product: Product) -> str:
    room = ROOMS.get(_kind(product), ROOMS["living"])
    dims = f" It measures {product.dimensions.label()}." if product.dimensions else ""
    return (
        f"Create a photorealistic lifestyle photograph of this exact {product.product_type} placed in {room.prompt}.{dims} "
        "Keep the product identical to the reference: same shape, proportions, colours, materials, finish, hardware "
        "and details. Do not add, remove or restyle any part of the product. Natural soft daylight, realistic scale "
        "and contact shadows, eye-level interior catalog photography, no people, no text, no logos, no watermark."
    )


def fetch_source(url: str, cache_dir: Path) -> Path:
    """Download the vendor photo once and keep it on disk (the vendor's S3 bucket can take 30s+ per image)."""
    path = cache_dir / "source_images" / url.rsplit("/", 1)[-1]
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _download(url)
    tmp = path.with_suffix(".part")
    tmp.write_bytes(data)
    tmp.replace(path)
    return path


@retry(retry=retry_if_exception_type((requests.RequestException,)), stop=stop_after_attempt(4),
       wait=wait_exponential(multiplier=2, min=2, max=20), reraise=True)
def _download(url: str) -> bytes:
    resp = requests.get(url, timeout=(10, 90), headers={"User-Agent": "Mozilla/5.0 (catalog-agent)"})
    resp.raise_for_status()
    if not resp.headers.get("content-type", "").startswith("image/"):
        raise ValueError(f"URL did not return an image (content-type {resp.headers.get('content-type')})")
    return resp.content


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def _gemini_retry(fn):
    from google.genai import errors

    @retry(retry=lambda rs: isinstance(rs.outcome.exception(), errors.APIError)
           and rs.outcome.exception().code in (500, 503, 504),
           stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=20), reraise=True)
    def run():
        return fn()

    return run()


# --- room scene renderer --------------------------------------------------------------------


@dataclass(frozen=True)
class Room:
    prompt: str
    decor: str  # tv | art | none
    rug: bool


ROOMS = {
    "tv": Room("a bright, modern living room against a painted wall, with a wall-mounted TV above it, a rug and a sofa edge in the foreground", "tv", True),
    "seating": Room("a cosy modern living room with a rug, a side table and framed art on the wall", "art", True),
    "table": Room("a modern living room on a large neutral rug, styled with a book and a small vase, sofa in soft focus behind", "art", True),
    "storage": Room("a calm, modern bedroom or dining room against a painted wall with framed art above", "art", False),
    "living": Room("a bright, modern living room", "art", True),
}

CAMERA_HEIGHT_IN = 40      # camera height above the floor
CAMERA_DISTANCE_IN = 150   # camera to the product's front edge
# The frame always shows at least this much room; coffee tables read better in a wider shot.
MIN_SCENE_WIDTH_IN = {"table": 120.0}
DEFAULT_SCENE_WIDTH_IN = 96.0


def _kind(p: Product) -> str:
    t = (p.product_type or "").lower()
    if "tv" in t or "media" in t:
        return "tv"
    if any(k in t for k in ("chair", "recliner", "sofa", "lounge", "chaise")):
        return "seating"
    if "table" in t:
        return "table"
    if any(k in t for k in ("dresser", "chest", "sideboard", "buffet", "cabinet", "console")):
        return "storage"
    return "living"


def render_room_scene(product: Product, cutout: Image.Image, seed: int | None = None) -> Image.Image:
    rng = random.Random(seed if seed is not None else product.sku)
    room = ROOMS[_kind(product)]
    dims = product.dimensions
    width_in = dims.width if dims else 48.0
    depth_in = dims.length if dims else 20.0

    # Scale: the cutout bbox spans the width plus a little of the depth in a 3/4 product view.
    # The product sets the scale (~56% of frame width, at most 58% of height); decor fits the space left.
    span_in = width_in + 0.25 * depth_in
    cut_aspect = cutout.height / cutout.width
    y_base = int(H * 0.90)
    min_scene = MIN_SCENE_WIDTH_IN.get(_kind(product), DEFAULT_SCENE_WIDTH_IN)
    visible_in = max(span_in / 0.56, min_scene)  # keeps small pieces small, real-world scale
    ppi = min(W / visible_in, 0.58 * H / (span_in * cut_aspect))
    prod_w = int(span_in * ppi)
    prod_h = int(prod_w * cut_aspect)

    # Perspective: f / z0 = ppi at the product's front plane; horizon from camera height.
    f = ppi * CAMERA_DISTANCE_IN
    y_horizon = y_base - CAMERA_HEIGHT_IN * ppi
    z_wall = CAMERA_DISTANCE_IN + depth_in + 3
    y_wall = int(y_horizon + f * CAMERA_HEIGHT_IN / z_wall)
    ppi_wall = f / z_wall

    palette = _palette(cutout, rng)
    scene = _wall(palette, y_wall, rng)
    _window_light(scene, y_wall)
    _floor(scene, palette, y_horizon, y_wall, f, rng)
    _baseboard(scene, y_wall, ppi_wall)
    x_center = W // 2 + int(rng.uniform(-0.04, 0.04) * W)
    if room.rug:
        _rug(scene, palette, f, y_horizon, x_center, width_in, depth_in, z_wall)

    prod = cutout.resize((prod_w, prod_h), Image.LANCZOS)
    x0, y0 = x_center - prod_w // 2, y_base - prod_h
    if room.decor == "tv":
        _wall_tv(scene, x_center, y0, ppi_wall, width_in)
    elif room.decor == "art":
        _wall_art(scene, palette, x_center, y0, ppi_wall, width_in, rng)

    scene = _vignette(scene)  # applied to the room only; product pixels stay untouched
    _shadows(scene, prod, x0, y0, y_wall)
    scene.paste(prod, (x0, y0), prod)
    return scene


def _palette(cutout: Image.Image, rng: random.Random) -> dict:
    arr = np.asarray(cutout.convert("RGBA"), dtype=np.float32)
    alpha = arr[..., 3] > 128
    lum = float((arr[..., :3][alpha] @ [0.299, 0.587, 0.114]).mean() / 255) if alpha.any() else 0.5
    if lum < 0.35:   # dark product -> light warm room
        walls = [(233, 227, 218), (226, 222, 214), (236, 231, 224)]
    elif lum > 0.68:  # light product -> mid-tone wall so it stands out
        walls = [(178, 186, 170), (196, 180, 163), (168, 180, 189), (190, 176, 158)]
    else:
        walls = [(238, 234, 228), (220, 214, 204), (205, 210, 200)]
    return {
        "wall": rng.choice(walls),
        "floor": rng.choice([(176, 136, 98), (160, 120, 86), (196, 160, 120)]),
        "rug": rng.choice([(222, 214, 200), (200, 196, 188), (214, 204, 186)]),
        "art": [rng.choice([(186, 120, 90), (120, 140, 120), (70, 90, 110), (210, 180, 120)]) for _ in range(3)],
    }


def _wall(p: dict, y_wall: int, rng: random.Random) -> Image.Image:
    base = np.array(p["wall"], dtype=np.float32)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    # Soft window light from the upper left, gentle falloff to the right and down.
    light = np.exp(-(((xx - 0.18 * W) / (0.55 * W)) ** 2 + ((yy - 0.25 * H) / (0.6 * H)) ** 2))
    shade = 0.90 + 0.14 * light - 0.05 * (yy / H)
    grain = np.asarray(Image.effect_noise((W, H), 6).filter(ImageFilter.GaussianBlur(1.2)), dtype=np.float32) - 128
    img = base[None, None, :] * shade[..., None] + grain[..., None] * 0.08
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), "RGB")


def _floor(scene: Image.Image, p: dict, y_h: float, y_wall: int, f: float, rng: random.Random) -> None:
    """Wood planks running away from the camera; lines converge on the vanishing point."""
    floor = Image.new("RGB", (W, H - y_wall), p["floor"])
    draw = ImageDraw.Draw(floor)
    vp_x = W / 2
    plank_in = 6.0
    for k in range(-60, 61):
        x_world = k * plank_in
        x_bottom = vp_x + f * x_world / (CAMERA_HEIGHT_IN * f / (H - y_h)) if H > y_h else vp_x
        x_top = vp_x + (x_bottom - vp_x) * (y_wall - y_h) / (H - y_h)
        x_bottom_next = vp_x + f * (x_world + plank_in) / (CAMERA_HEIGHT_IN * f / (H - y_h))
        x_top_next = vp_x + (x_bottom_next - vp_x) * (y_wall - y_h) / (H - y_h)
        tone = rng.uniform(0.88, 1.08)
        color = tuple(int(min(255, c * tone)) for c in p["floor"])
        draw.polygon([(x_top, 0), (x_top_next, 0), (x_bottom_next, H - y_wall), (x_bottom, H - y_wall)], fill=color)
        draw.line([(x_top, 0), (x_bottom, H - y_wall)], fill=tuple(int(c * 0.72) for c in p["floor"]), width=1)
    arr = np.asarray(floor, dtype=np.float32)
    yy = np.linspace(0, 1, floor.height)[:, None, None]
    xx = np.linspace(0, 1, W)[None, :, None]
    arr = arr * (0.82 + 0.22 * yy) * (1.02 - 0.06 * np.abs(xx - 0.35))
    grain = np.asarray(Image.effect_noise((W, floor.height), 14).filter(ImageFilter.BoxBlur(1)), dtype=np.float32) - 128
    arr += grain[..., None] * 0.10
    scene.paste(Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)), (0, y_wall))


def _baseboard(scene: Image.Image, y_wall: int, ppi_wall: float) -> None:
    h = max(6, int(4.5 * ppi_wall))
    draw = ImageDraw.Draw(scene)
    draw.rectangle([0, y_wall - h, W, y_wall], fill=(244, 242, 238))
    draw.line([(0, y_wall - h), (W, y_wall - h)], fill=(214, 210, 204), width=2)
    draw.line([(0, y_wall), (W, y_wall)], fill=(150, 140, 130), width=2)


def _rug(scene: Image.Image, p: dict, f: float, y_h: float, xc: int, width_in: float, depth_in: float, z_wall: float) -> None:
    rug_w, front, back = max(78.0, width_in * 1.35), -30.0, z_wall - CAMERA_DISTANCE_IN - 2.0

    def project(x_in: float, z_off: float) -> tuple[float, float]:
        z = CAMERA_DISTANCE_IN + z_off
        return xc + f * x_in / z, y_h + f * CAMERA_HEIGHT_IN / z

    corners = [project(-rug_w / 2, back), project(rug_w / 2, back), project(rug_w / 2, front), project(-rug_w / 2, front)]
    layer = Image.new("RGBA", scene.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.polygon(corners, fill=(*p["rug"], 255))
    for inset_in, shade in ((5, 0.80), (8, 0.88)):
        inset = [project(-rug_w / 2 + inset_in, back - inset_in * 0.6), project(rug_w / 2 - inset_in, back - inset_in * 0.6),
                 project(rug_w / 2 - inset_in, front + inset_in), project(-rug_w / 2 + inset_in, front + inset_in)]
        d.line([*inset, inset[0]], fill=(*[int(c * shade) for c in p["rug"]], 255), width=4)
    texture = Image.effect_noise(scene.size, 18).convert("L").point(lambda v: 128 + (v - 128) // 4)
    rug_rgb = Image.composite(Image.merge("RGB", [texture] * 3), layer.convert("RGB"), Image.new("L", scene.size, 40))
    scene.paste(rug_rgb, (0, 0), layer.split()[3])


def _wall_tv(scene: Image.Image, xc: int, prod_top: int, ppi: float, stand_w: float) -> None:
    bottom = prod_top - int(7 * ppi)
    available_in = (bottom - H * 0.05) / ppi
    tv_w_in = min(65.0, stand_w * 0.85, available_in * 16 / 9)
    if tv_w_in < 32:
        return  # not enough wall above the stand for a believable TV
    tw, th = int(tv_w_in * ppi), int(tv_w_in * 9 / 16 * ppi)
    top = bottom - th
    box = [xc - tw // 2, top, xc + tw // 2, bottom]
    _soft_shadow(scene, box, radius=18, opacity=60, offset=(10, 12))
    d = ImageDraw.Draw(scene)
    d.rounded_rectangle(box, radius=6, fill=(22, 22, 24))
    inner = [box[0] + 6, box[1] + 6, box[2] - 6, box[3] - 6]
    screen = Image.linear_gradient("L").resize((inner[2] - inner[0], inner[3] - inner[1])).rotate(35, expand=False)
    screen_rgb = Image.merge("RGB", [screen.point(lambda v: 18 + v // 9)] * 3)
    scene.paste(screen_rgb, (inner[0], inner[1]))


def _wall_art(scene: Image.Image, p: dict, xc: int, prod_top: int, ppi: float, prod_w_in: float, rng: random.Random) -> None:
    bottom = prod_top - int(8 * ppi)
    available_in = (bottom - H * 0.06) / ppi
    art_w_in = min(40.0, max(24.0, prod_w_in * 0.6), available_in / 0.72)
    if art_w_in < 16:
        return  # no room above the product
    aw, ah = int(art_w_in * ppi), int(art_w_in * 0.72 * ppi)
    top = bottom - ah
    box = [xc - aw // 2, top, xc + aw // 2, bottom]
    _soft_shadow(scene, box, radius=14, opacity=55, offset=(8, 10))
    d = ImageDraw.Draw(scene)
    d.rectangle(box, fill=(40, 36, 32))
    mat = [box[0] + 8, box[1] + 8, box[2] - 8, box[3] - 8]
    d.rectangle(mat, fill=(246, 243, 236))
    art = [mat[0] + int(0.12 * aw), mat[1] + int(0.12 * ah), mat[2] - int(0.12 * aw), mat[3] - int(0.12 * ah)]
    # Simple abstract composition: three soft colour fields.
    aw2 = art[2] - art[0]
    cuts = sorted([rng.uniform(0.25, 0.45), rng.uniform(0.55, 0.8)])
    xs = [art[0], art[0] + int(aw2 * cuts[0]), art[0] + int(aw2 * cuts[1]), art[2]]
    for i, color in enumerate(p["art"]):
        d.rectangle([xs[i], art[1], xs[i + 1], art[3]], fill=color)


def _soft_shadow(scene: Image.Image, box: list[int], radius: int, opacity: int, offset: tuple[int, int]) -> None:
    mask = Image.new("L", scene.size, 0)
    ImageDraw.Draw(mask).rectangle([box[0] + offset[0], box[1] + offset[1], box[2] + offset[0], box[3] + offset[1]], fill=opacity)
    mask = mask.filter(ImageFilter.GaussianBlur(radius))
    scene.paste(Image.new("RGB", scene.size, (40, 34, 28)), (0, 0), mask)


def _shadows(scene: Image.Image, prod: Image.Image, x0: int, y0: int, y_wall: int) -> None:
    alpha = prod.split()[3]
    w, h = prod.size
    shadow_rgb = Image.new("RGB", scene.size, (28, 22, 16))

    # Footprint: the bottom band of the silhouette is where the product meets the floor.
    band = max(6, int(h * 0.10))
    footprint = Image.new("L", (w, h), 0)
    footprint.paste(alpha.crop((0, h - band, w, h)), (0, h - band))

    # Contact shadow: footprint pushed down a few px and tightly blurred, so the base sits on the floor.
    contact = Image.new("L", scene.size, 0)
    contact.paste(footprint.point(lambda v: int(v * 0.85)), (x0, y0 + 4))
    contact = contact.filter(ImageFilter.GaussianBlur(6))

    # Cast shadow: light from the front-left throws the silhouette back toward the wall (up-right in the image).
    cast_h = max(10, int(h * 0.22))
    cast = Image.new("L", scene.size, 0)
    squashed = alpha.resize((w, cast_h)).point(lambda v: int(v * 0.42))
    cast.paste(squashed, (x0 + int(w * 0.06), y0 + h - cast_h - int(h * 0.02)))
    cast = cast.filter(ImageFilter.GaussianBlur(max(12, w // 30)))

    # Wall shadow: soft, offset right, only above the skirting line.
    wall = Image.new("L", scene.size, 0)
    wall.paste(alpha.point(lambda v: int(v * 0.20)), (x0 + int(w * 0.05), y0 - int(h * 0.03)))
    wall = wall.filter(ImageFilter.GaussianBlur(max(16, w // 22)))
    wall.paste(0, (0, y_wall, W, H))

    for m in (wall, cast, contact):
        scene.paste(shadow_rgb, (0, 0), m)


def _window_light(scene: Image.Image, y_wall: int) -> None:
    """Two soft, skewed panes of daylight on the wall: reads as a window just out of frame."""
    mask = Image.new("L", scene.size, 0)
    d = ImageDraw.Draw(mask)
    for x, wpx in ((int(W * 0.06), int(W * 0.11)), (int(W * 0.19), int(W * 0.11))):
        d.polygon([(x, int(H * 0.10)), (x + wpx, int(H * 0.08)), (x + wpx + int(W * 0.10), y_wall), (x + int(W * 0.10), y_wall)], fill=34)
    mask = mask.filter(ImageFilter.GaussianBlur(28))
    scene.paste(Image.new("RGB", scene.size, (255, 250, 238)), (0, 0), mask)


def _vignette(scene: Image.Image) -> Image.Image:
    arr = np.asarray(scene, dtype=np.float32)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    vignette = 1 - 0.18 * (((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2) / 2
    arr *= vignette[..., None]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
