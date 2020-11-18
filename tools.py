# Copyright 2018-2020 Ivan Alles. See also the LICENSE file.

import os

import train

if __name__ == '__main__':
    trainer = train.Trainer(os.path.join(r'data\models\tools\config.json'))
    trainer.run()



