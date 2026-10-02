# import json

# from ai_companion.modules.transport.tools import plan_journey

# print(json.dumps(plan_journey.args_schema.model_json_schema(), indent=2))
# print(plan_journey.invoke(
#     {"origin": "Alexanderplatz, Berlin", "destination": "Flughafen BER, Berlin"}))


from ai_companion.modules.transport.tools import _find_place

for query in ["Alexanderplatz, Berlin", "Flughafen BER, Berlin", "Kreuzberg, Berlin", "xqzzvy"]:
    place = _find_place(query)
    print(query, "→", place and (
        place["type"], place["name"], place["lat"], place["lon"]))
