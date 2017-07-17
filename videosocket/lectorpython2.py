import cv
if __name__ == '__main__':
  capture = cv.CreateFileCapture('video.fifo')
  loop = True
  while(loop):
    frame = cv.QueryFrame(capture)
    if (frame == None):
            break;
    cv.ShowImage('Wild Life', frame)
    char = cv.WaitKey(33)
    if (char != -1):
        if (ord(char) == 27):
            loop = False

