"""Fetch the pinned public MaintIE corpus and verify the audited checksum."""
import hashlib
import urllib.request
from pathlib import Path

REVISION = "825e7710e4129598da997276a38b3cb40b200bc2"
DATA_SHA256 = "ef811cf4ffd0c90e6a13b1369e24e9d1d08f9a294d873ab4fb1a580096d7d6e0"
BASE_URL = f"https://raw.githubusercontent.com/nlp-tlp/maintie/{REVISION}"


def main():
    directory = Path("data/raw")
    directory.mkdir(parents=True, exist_ok=True)
    for name in ["gold_release.json", "LICENSE.md"]:
        destination = directory / name
        prefix = "data/" if name.endswith(".json") else ""
        with urllib.request.urlopen(f"{BASE_URL}/{prefix}{name}", timeout=60) as response:
            content = response.read()
        if name == "gold_release.json" and hashlib.sha256(content).hexdigest() != DATA_SHA256:
            raise ValueError("Downloaded corpus differs from the audited version.")
        if destination.exists() and destination.read_bytes() != content:
            raise FileExistsError(f"Refusing to replace different local content: {destination}")
        destination.write_bytes(content)
        print(f"Verified download: {destination}")


if __name__ == "__main__":
    main()
