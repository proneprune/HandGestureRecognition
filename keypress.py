from collections import deque
import ctypes
import ctypes.wintypes


KEYEVENTF_KEYDOWN = 0x0000  #PYAutogui
KEYEVENTF_KEYUP = 0x0002    #PYAutogui


def copy():
    ctypes.windll.user32.keybd_event(0x0c, 0, KEYEVENTF_KEYDOWN, 0) #ctrl
    ctypes.windll.user32.keybd_event(0x43, 0, KEYEVENTF_KEYDOWN, 0) #c

    ctypes.windll.user32.keybd_event(0x43, 0, KEYEVENTF_KEYUP, 0) #c
    ctypes.windll.user32.keybd_event(0x0c, 0, KEYEVENTF_KEYUP, 0) #ctrl

def paste():
    ctypes.windll.user32.keybd_event(0x0c, 0, KEYEVENTF_KEYDOWN, 0) #ctrl
    ctypes.windll.user32.keybd_event(0x56, 0, KEYEVENTF_KEYDOWN, 0) #v

    ctypes.windll.user32.keybd_event(0x56, 0, KEYEVENTF_KEYUP, 0) #v
    ctypes.windll.user32.keybd_event(0x0c, 0, KEYEVENTF_KEYUP, 0) #ctrl

def back():
    ctypes.windll.user32.keybd_event(0x12, 0, KEYEVENTF_KEYDOWN, 0) #ALT
    ctypes.windll.user32.keybd_event(0x25, 0, KEYEVENTF_KEYDOWN, 0) #left arrow

    ctypes.windll.user32.keybd_event(0x25, 0, KEYEVENTF_KEYUP, 0) #left arrow
    ctypes.windll.user32.keybd_event(0x12, 0, KEYEVENTF_KEYUP, 0) #ALT 


def escape():
    ctypes.windll.user32.keybd_event(0x1b, 0, KEYEVENTF_KEYDOWN, 0) #esc
    ctypes.windll.user32.keybd_event(0x1b, 0, KEYEVENTF_KEYUP, 0) #esc

def lowervolume():              
    ctypes.windll.user32.keybd_event(0xae, 0, KEYEVENTF_KEYDOWN, 0) #volumedown
    ctypes.windll.user32.keybd_event(0xae, 0, KEYEVENTF_KEYUP, 0) #volumedown

def raisevolume():
    ctypes.windll.user32.keybd_event(0xaf, 0, KEYEVENTF_KEYDOWN, 0) #volumeup
    ctypes.windll.user32.keybd_event(0xaf, 0, KEYEVENTF_KEYUP, 0) #volumeup

def linkedin():
                                
    ctypes.windll.user32.keybd_event(0x11, 0, KEYEVENTF_KEYDOWN, 0) #ctrl
    ctypes.windll.user32.keybd_event(0x12, 0, KEYEVENTF_KEYDOWN, 0) #alt
    ctypes.windll.user32.keybd_event(0x10, 0, KEYEVENTF_KEYDOWN, 0) #shift
    ctypes.windll.user32.keybd_event(0x5b, 0, KEYEVENTF_KEYDOWN, 0) #win
    ctypes.windll.user32.keybd_event(0x4C, 0, KEYEVENTF_KEYDOWN, 0) #L

    ctypes.windll.user32.keybd_event(0x4C, 0, KEYEVENTF_KEYUP, 0) #minus
    ctypes.windll.user32.keybd_event(0x5b, 0, KEYEVENTF_KEYUP, 0) #win 
    ctypes.windll.user32.keybd_event(0x10, 0, KEYEVENTF_KEYUP, 0) #shift
    ctypes.windll.user32.keybd_event(0x12, 0, KEYEVENTF_KEYUP, 0) #alt
    ctypes.windll.user32.keybd_event(0x11, 0, KEYEVENTF_KEYUP, 0) #ctrl 

def mute():
    ctypes.windll.user32.keybd_event(0xAD, 0, KEYEVENTF_KEYDOWN, 0) #mute
    ctypes.windll.user32.keybd_event(0xAD, 0, KEYEVENTF_KEYUP, 0) #mute

def zoomout():
    ctypes.windll.user32.keybd_event(0x11, 0, KEYEVENTF_KEYDOWN, 0) #ctrl
    ctypes.windll.user32.keybd_event(0x6D, 0, KEYEVENTF_KEYDOWN, 0) #minus

    ctypes.windll.user32.keybd_event(0x6D, 0, KEYEVENTF_KEYUP, 0) #minus
    ctypes.windll.user32.keybd_event(0x11, 0, KEYEVENTF_KEYUP, 0) #ctrl

def zoomin():
    ctypes.windll.user32.keybd_event(0x11, 0, KEYEVENTF_KEYDOWN, 0) #ctrl
    ctypes.windll.user32.keybd_event(0x6B, 0, KEYEVENTF_KEYDOWN, 0) #plus

    ctypes.windll.user32.keybd_event(0x6B, 0, KEYEVENTF_KEYUP, 0) #plus
    ctypes.windll.user32.keybd_event(0x11, 0, KEYEVENTF_KEYUP, 0) #ctrl

def alttab():
    ctypes.windll.user32.keybd_event(0x12, 0, KEYEVENTF_KEYDOWN, 0) #alt
    ctypes.windll.user32.keybd_event(0x09, 0, KEYEVENTF_KEYDOWN, 0) #tab

    ctypes.windll.user32.keybd_event(0x09, 0, KEYEVENTF_KEYUP, 0) #tab
    ctypes.windll.user32.keybd_event(0x12, 0, KEYEVENTF_KEYUP, 0) #alt          


def skelettinit():
    skelett_history = deque(maxlen=16)

    for i in range(16):
        skelett_history.append([[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]])

    return skelett_history