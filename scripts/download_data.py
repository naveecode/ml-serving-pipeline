"""Download the IBM telco churn CSV if it is not already present."""

from pathlib import Path
from urllib.request import urlopen

URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
DESTINATION = Path(__file__).resolve().parents[1] / "data" / "Telco-Customer-Churn.csv"


def main() -> None:
    if DESTINATION.exists() and DESTINATION.stat().st_size > 0:
        print(f"already present: {DESTINATION}")
        return
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(URL, timeout=60) as response:
        DESTINATION.write_bytes(response.read())
    print(f"wrote {DESTINATION} ({DESTINATION.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
