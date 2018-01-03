import os
import time

from datetime import datetime
from multiprocessing import Process


def timer(start=None, to_var=False):
    if not start:
        print(datetime.now().ctime())
        return time.time()
    stop = time.time()
    m, s = divmod(stop - start, 60)
    h, m = divmod(m, 60)
    if to_var:
        return '{}:{}:{}'.format(int(h), int(m), round(s))
    print('total time {}:{}:{}'.format(int(h), int(m), round(s)))


def format_ratings_file(filename, delim, train=True):
    list_filename = filename.split("/")
    path, old_filename = '/'.join(list_filename[:-1]), list_filename[-1]
    new_filename = "train_" if train else "test_"
    new_filename += old_filename.split(".")[0] + ".csv"
    new_filename = path + "/" + new_filename
    os.system(
              ("sed 's/{old_delim}/,/g' {filename} > {new_filename}"
               ).format(
                        old_delim=delim,
                        filename=filename,
                        new_filename=new_filename
                        )
              )
    with open(new_filename, 'r') as original:
        data = original.read()
    with open(new_filename, 'w') as modified:
        modified.write("user,item,rating\n" + data)


class BaseMultiprocessing(object):
    def __init__(self, max_active_processes=10):
        """Init."""
        self.active_process_list = []
        self.max_active_processes = max_active_processes
        self.current_active_processes = 0

    def new_process(self, func, **kwargs):
        """Spawn new process."""
        p = Process(target=func, kwargs=kwargs)
        p.start()
        self.active_process_list.append(p)
        self.current_active_processes += 1

    def synchronize(self):
        """Wait for all the alive process to terminate."""
        for process in self.active_process_list:
            process.join()
        self.active_process_list = []
        self.current_active_processes = 0
