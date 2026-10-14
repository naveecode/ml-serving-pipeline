"""Load test for the scoring API. Point --host at a running server."""

from locust import HttpUser, between, task

from churn.samples import EXAMPLE_CUSTOMER

_LONG_TENURE = dict(EXAMPLE_CUSTOMER)
_LONG_TENURE["tenure"] = 72
_LONG_TENURE["Contract"] = "Two year"
_LONG_TENURE["MonthlyCharges"] = 85.5
_LONG_TENURE["TotalCharges"] = 6156.0


class ChurnUser(HttpUser):
    wait_time = between(0.05, 0.3)

    @task(6)
    def predict(self) -> None:
        self.client.post("/predict", json=EXAMPLE_CUSTOMER)

    @task(2)
    def predict_batch(self) -> None:
        self.client.post(
            "/predict/batch",
            json={"records": [EXAMPLE_CUSTOMER, _LONG_TENURE]},
        )

    @task(1)
    def health(self) -> None:
        self.client.get("/health")
