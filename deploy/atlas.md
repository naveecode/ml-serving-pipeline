# MongoDB Atlas

The API writes every `/predict` and `/predict/batch` call to the `predictions` collection. Atlas is the hosted MongoDB the Render and Fly configs expect.

1. Create an M0 (or larger) cluster in MongoDB Atlas.
2. Database Access: create a database user with read/write on the `churn` database.
3. Network Access: allow the host that will run the API. Render and Fly egress IPs change, so a temporary `0.0.0.0/0` rule is the usual demo setting. Restrict it if you have fixed egress.
4. Connect: copy the SRV string and set it as `MONGODB_URI`.

```text
MONGODB_URI=mongodb+srv://USER:PASSWORD@CLUSTER.mongodb.net/?retryWrites=true&w=majority
MONGODB_DB=churn
MONGODB_COLLECTION=predictions
```

The collection is created on the first scored request. No schema migration step.

Locally, `docker compose up` provides MongoDB instead, and `MONGODB_URI` is `mongodb://mongo:27017` inside the compose network.

On Kubernetes, the `churn-mongodb` Secret key `uri` holds the same string. The base manifest ships a non-routable placeholder (`mongodb://mongodb:27017`). Replace that value before applying the manifests to a cluster that should log to Atlas.
