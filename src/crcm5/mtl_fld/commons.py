from pathlib import Path


img_folder = Path("img_mtl_flood_2017")


if not img_folder.exists():
    img_folder.mkdir()