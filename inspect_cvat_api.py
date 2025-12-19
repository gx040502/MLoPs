from cvat_sdk import make_client
import inspect

url = "http://100.110.21.83:8080"
username = "gyapxun"
password = "mmcgtm2014"

with make_client(url, credentials=(username, password)) as client:
    print("Connected.")
    print("\n--- Tasks API Methods ---")
    for name, method in inspect.getmembers(client.tasks_api, predicate=inspect.ismethod):
        print(name)

    print("\n--- Requests API Methods ---")
    for name, method in inspect.getmembers(client.requests_api, predicate=inspect.ismethod):
        print(name)
