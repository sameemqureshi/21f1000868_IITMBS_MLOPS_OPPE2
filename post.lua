wrk.method = "POST"
wrk.headers["Content-Type"] = "application/json"
wrk.body = io.open("artifacts/random_100_payload.json", "r"):read("*a")
