# Copyright 2018-2020 Ivan Alles. See also the LICENSE file.

import os

from localizer import train

if __name__ == '__main__':
    trainer = train.Trainer(os.path.join(os.path.dirname(__file__), 'config.json'))
    trainer.run()



