import subprocess, time, threading
from typing import List, Union
import os, shutil, fnmatch

class Channel:
    def __init__(self,source,destination,sync_deletions=False,every=60,exclude: Union[str, List, None] = None):
        self.source = source
        self.destination = destination
        self.event = threading.Event()
        self.syncing_thread = threading.Thread(target=self._sync,args=())
        self.sync_deletions = sync_deletions
        self.every = every
        if not exclude:
            exclude = []
        if isinstance(exclude,str):
            exclude = [exclude]
        self.exclude = exclude
        self.command = ['rsync','-aP']

    def alive(self):#Check if the thread is alive
        if self.syncing_thread.is_alive():
            return True
        else:
            return False

    def _sync(self):#Sync constantly
        command = self.command
        for exclusion in self.exclude:
            command.append(f'--exclude={exclusion}')
        command.extend([f'{self.source}/',f'{self.destination}/'])
        if self.sync_deletions:    
            command.append('--delete')
        while not self.event.is_set():
            subprocess.run(command)
            time.sleep(self.every)

    def copy(self):#Sync once
        command = self.command
        for exclusion in self.exclude:
            command.append(f'--exclude={exclusion}')
        command.extend([f'{self.source}/',f'{self.destination}/'])
        if self.sync_deletions:    
            command.append('--delete')
        subprocess.run(command)
        return True
    
    def start(self):#Handle threads
        if self.syncing_thread.is_alive():#Check if it's running
            self.event.set()
            self.syncing_thread.join()
        if self.event.is_set():
            self.event.clear()
        if self.syncing_thread._started.is_set():#If it has been started before
            self.syncing_thread = threading.Thread(target=self._sync,args=())#Create a FRESH thread
        self.syncing_thread.start()#Start the thread
        return self.alive()

    def stop(self):#Stop the thread and close the process
        if self.alive():
            self.event.set()
            self.syncing_thread.join()
            while self.alive():
                if not self.alive():
                    break
        return not self.alive()
    
class GarbageMan:
    def __init__(self) -> None:
        self.thread = threading.Thread(target=self.take_out,args=())
        self.event = threading.Event()

    def destroy(self, trash):
        if not isinstance(trash,dict):
            if os.path.isdir(os.path.join(self.path,trash)):
                shutil.rmtree(os.path.join(self.path,trash))
            elif os.path.isfile(os.path.join(self.path,trash)):
                os.remove(os.path.join(self.path,trash))
        else:
            trash.Delete()

    def take_out(self) -> None:
        while not self.event.is_set():
            for object in self.garbage:
                trash = object["title"] if isinstance(object,dict) else object
                if fnmatch.fnmatch(trash,self.pattern):
                    self.destroy(object)
            time.sleep(self.every)

    def stop(self) -> None:
        if not self.event.is_set():
            self.event.set()
            self.thread.join()
        self.event.clear()
        if self.thread._started.is_set():
            self.thread = threading.Thread(target=self.take_out,args=())

    def start(self,path: Union[str,List],every:int=30,pattern: str='') -> None:
        if isinstance(path,list):
            self.path = None
            self.garbage = path
        elif isinstance(path,str):
            self.path = path
            self.garbage = os.listdir(path)
        else:
            return "Error"
        self.every = every
        self.pattern = pattern
        if self.thread.is_alive():
            self.stop()
        self.thread.start()

    def _fake(self, trash):
        if not isinstance(trash,dict):
            if os.path.isdir(os.path.join(self.path,trash)):
                with open("log.txt","a") as f:
                    f.write(f"Fake deleted dir: {trash}")
            elif os.path.isfile(os.path.join(self.path,trash)):
                with open("log.txt","a") as f:
                    f.write(f"Fake deleted file: {trash}")
        else:
            with open("log.txt","a") as f:
                    f.write(f"Fake permanently deleted: {trash['title']}")