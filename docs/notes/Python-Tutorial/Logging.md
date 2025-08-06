# Logging

```python
import logging

logging.basicConfig(level=logging.INFO) #只有当日志等级高于此时才会打印出来

logging.debug("debug msg")
logging.info("info msg")
logging.warning("warning msg")
logging.error("error msg")
logging.critical("critical msg")
```

默认输出：日志等级+日志名称+日志内容

## 参考

Logging format: [https://docs.python.org/3/library/logging.html#logrecord-attributes](https://docs.python.org/3/library/logging.html#logrecord-attributes)

Logging handler: [https://docs.python.org/3/howto/logging.html#useful-handlers](https://docs.python.org/3/howto/logging.html#useful-handlers)

`logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(name)s - %(levelname)s - %(message)s)`

添加filename='?.txt'

filemode = 'w' / 'a'

`logger = logging.getLogger('test_logger')` 
or `logging.getlogger(__name__) #可能是__main__或者module.py`

## 异常处理

```python

try:
    1 / 0
except:
    test_logger.exception("Get exception")
```
