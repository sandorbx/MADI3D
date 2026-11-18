"""Bounded annotation-only transfers from verified official static sources."""
from pathlib import Path
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone
import shutil
from tempfile import TemporaryDirectory
from time import monotonic
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


ANNOTATION_PRODUCTS = ("consolidated_cell_types", "classification", "neurons", "labels")
MANC_ANNOTATION_PRODUCT = "segment_properties"
ARCHIVE = "https://storage.googleapis.com/flywire-data/codex/data/fafb/783/"
CODEX_RESOURCE = "https://codex.flywire.ai/api/download_resource"
CODEX_ABOUT = "https://codex.flywire.ai/about_flywire"
FLYWIRE_GUIDELINES = "https://home.flywire.ai/guidelines"


class _StaticRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        target = urlsplit(newurl)
        if (target.scheme != "https" or target.hostname not in
                {"codex.flywire.ai", "storage.googleapis.com"} or
                target.username or target.password or "api_token=" in target.query):
            raise ValueError("Unsupported static-product redirect.")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download_and_install(catalog, products=None, *, token=None,
                         cancelled=lambda: False, progress=lambda text: None):
    """A token selects the FAQ-documented current consolidated-cell-type product.

    FAFB products use the verified public m783 archive; MANC uses its exact public
    segment-properties object; MCNS uses its curated Feather product. Tokens are used only in
    memory for the Codex request, never in catalog evidence, filenames or errors.
    The authenticated manifest is not inferred from the public archive listing.
    """
    from .catalog import CatalogProduct, MCNS_ANNOTATION_PRODUCT

    if products is None:
        products = ((MCNS_ANNOTATION_PRODUCT,) if catalog.context.dataset == "mcns" else
                    (MANC_ANNOTATION_PRODUCT,) if catalog.context.dataset == "manc" else ANNOTATION_PRODUCTS)
    products = tuple(products)
    is_manc = catalog.context.dataset == "manc"
    is_mcns = catalog.context.dataset == "mcns"
    # Public setup installs only curated MCNS annotations, without the separate
    # transmitter-prediction table. Both can be installed explicitly from files.
    supported_products = {MCNS_ANNOTATION_PRODUCT} if is_mcns else {MANC_ANNOTATION_PRODUCT} if is_manc else set(ANNOTATION_PRODUCTS)
    if not products or len(set(products)) != len(products) or set(products) - supported_products:
        raise ValueError("Choose supported annotation products.")
    if token is not None and (catalog.context.dataset != "fafb" or products != ("consolidated_cell_types",)):
        raise ValueError("The Codex token route supports consolidated cell types only.")
    catalog.path.parent.mkdir(parents=True, exist_ok=True)
    opener = build_opener(_StaticRedirects())
    deadline = monotonic() + 180
    with TemporaryDirectory(prefix=".flywire-download-", dir=catalog.path.parent) as folder:
        sources = []
        for product in products:
            if cancelled():
                raise ValueError("Catalog update cancelled.")
            progress(f"Downloading {product}")
            url = catalog.context.profile.annotation_source if is_manc or is_mcns else ARCHIVE + product + ".csv.gz"
            if token is not None:
                url = CODEX_RESOURCE + "?" + urlencode({
                    "dataset": "fafb", "data_product": product, "api_token": token})
            path = Path(folder) / (product + (".feather" if is_mcns else ".json" if is_manc else ".csv.gz"))
            source_locator = url
            if token is not None:
                source_locator = CODEX_RESOURCE + "?" + urlencode({
                    "dataset": "fafb", "data_product": product})
            acquired_at_utc = None
            try:
                with opener.open(Request(url), timeout=15) as response, path.open("wb") as output:
                    stamp = response.headers.get("Last-Modified")
                    date = parsedate_to_datetime(stamp).isoformat() if stamp else None
                    size = 0
                    while True:
                        if cancelled():
                            raise ValueError("Catalog update cancelled.")
                        if monotonic() > deadline:
                            raise ValueError("Catalog download deadline exceeded.")
                        # HTTPResponse.read1 returns after one socket read. A full
                        # read(n) can wait indefinitely on a trickling server.
                        block = response.read1(64 * 1024)
                        if not block:
                            break
                        size += len(block)
                        if size > catalog.limits.max_compressed_bytes:
                            raise ValueError("Annotation product exceeds the download limit.")
                        if shutil.disk_usage(folder).free < len(block) + catalog.limits.reserve_disk_bytes:
                            raise ValueError("Insufficient disk space for annotation download.")
                        output.write(block)
                acquired_at_utc = datetime.now(timezone.utc).isoformat()
            except Exception:
                # HTTP/URL exceptions can contain the complete credential URL.
                local_format = "Feather product" if is_mcns else "properties JSON" if is_manc else "CSV.gz"
                message = "Catalog update cancelled." if cancelled() else f"Static annotation download failed. Check access and retry, or import a local {local_format}."
                raise ValueError(message) from None
            sources.append(CatalogProduct(
                path, product, source_date=date,
                acquired_at_utc=acquired_at_utc, source_locator=source_locator,
                citation_urls=(CODEX_ABOUT, FLYWIRE_GUIDELINES) if catalog.context.dataset == "fafb" else None,
            ))
        return catalog.install(sources, cancelled=cancelled, progress=progress)
