from cvat_sdk.api_client import Configuration, ApiClient
import inspect

url = "http://100.110.21.83:8080"
username = "gyapxun"
password = "mmcgtm2014"

config = Configuration(host=url, username=username, password=password)
client = ApiClient(config)

print("\n--- ApiClient Attributes ---")
print([a for a in dir(client) if not a.startswith('_')])

if hasattr(client, 'tasks_api'):
    print("\n--- TasksApi Methods ---")
    # Assuming it's a property or object
    tasks_api = client.tasks_api
    for name, method in inspect.getmembers(tasks_api, predicate=inspect.ismethod):
        print(name)
else:
    print("\nApiClient has no 'tasks_api' attribute.")
