import os
import re
import sys
import logging
import colorlog

script_path = os.path.abspath(__file__)
package_path = re.findall(".*/baikal", script_path)[0]
dir = os.path.dirname(package_path)
# print(dir)
sys.path.append(dir)

from progress_notify import ProgressStream  # noqa: E402


# set success level
logging.SUCCESS = 25
logging.addLevelName(logging.SUCCESS, "SUCCESS")


def success_func(self, message, *args, **kws):
    self._log(logging.SUCCESS, message, args, **kws)


logging.Logger.success = success_func

# use root logger below
# format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
# logging.basicConfig(format=format, level=logging.DEBUG)

handler = logging.StreamHandler(ProgressStream())
handler.setLevel(logging.DEBUG)
handler.setFormatter(
    colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s | %(levelname)s | %(name)s-%(process)d | %(message)s",
        # datefmt="%Y-%d-%d %H:%M:%S",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "blue",
            "SUCCESS": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bgred",
        },
    )
)

logger = logging.getLogger()
logger.addHandler(handler)


logger.setLevel(logging.DEBUG)

logging.getLogger("matplotlib").setLevel(logging.INFO)

if __name__ == "__main__":
    logger.success("success!!!")
    logger.info("info!!!")
    logger.error("error!!!")
