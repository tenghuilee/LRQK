#! python

from opencompass.registry import TASKS, MODELS, RUNNERS
from lrqk.wrap.chat_lrqk import LRQKChatBot
from lrqk.wrap.chat_kvquant import KVQuantChatBot
import lrqk.wrap.local 
import lrqk.wrap.openicl_infer
import lrqk.wrap.openicl_eval


if __name__ == '__main__':
    from opencompass.cli.main import main
    print(RUNNERS)
    main()

