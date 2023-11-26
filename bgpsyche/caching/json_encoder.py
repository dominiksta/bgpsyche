from datetime import datetime
import json


def _datetime_valid(dt_str: str):
    try:
        datetime.fromisoformat(dt_str)
    except:
        return False
    return True


class JSONDatetimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime): return obj.isoformat()
        return json.JSONEncoder.default(self, obj)


class JSONDatetimeDecoder(json.JSONDecoder):
    def __init__(self, **kwargs):
        kwargs['object_hook'] = self.object_hook
        super().__init__(**kwargs)

    def object_hook(self, obj: dict):
        for key, value in obj.items():
            if isinstance(value, str) and _datetime_valid(value):
                obj[key] = datetime.fromisoformat(value)
        return obj



if __name__ == '__main__':
    print(json.dumps({'hi': datetime.now()}, cls=JSONDatetimeEncoder))
    print(json.loads('''{
      "hi": "2023-08-18T20:13:57.627517",
      "yes": 4,
      "very": "haha"
    }''', cls=JSONDatetimeDecoder))
    print(json.loads('4', cls=JSONDatetimeDecoder))
    print(json.loads('"hi"', cls=JSONDatetimeDecoder))
    # print(json.loads('', cls=JSONDatetimeDecoder))