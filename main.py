
import cv2 as cv
import numpy as np
import mediapipe as mp
import curses
import os
# import rembg # not using as it is slow


WINDOWNAME = "camera"
# https://paulbourke.net/dataformats/asciiart/
ASCII  = " @%#*+=-:." 
COUNT = len(ASCII)-1


mp_selfie_segmentation = mp.solutions.selfie_segmentation
mp_drawing = mp.solutions.drawing_utils

def DisplayASCII(stdscr, frame):
    stdscr.clear()  # Clear the screen
    row = frame.shape[0]
    column = frame.shape[1]

    # print("shape:",row,column)

    # Create a list to hold the ASCII art lines
    ascii_art = ""

    for i in range(row):
        # Loop through each column of the current row
        for j in range(column):
            # print(i,j,frame[i][j])
            ascii_art += ASCII[frame[i][j]]  # Get corresponding ASCII character
        ascii_art += "\n"  # Add newline after each row

    # Add the entire ASCII art at once to the screen
    stdscr.addstr(ascii_art)  
    stdscr.refresh()  # Refresh the screen to display the updated content

def Clear():
    # Check the operating system
    if os.name == 'nt':  # For Windows
        os.system('cls')
    else:  # For Unix-based systems (Linux, macOS)
        os.system('clear')

def DisplayASCII2(frame):
    Clear()
    row = frame.shape[0]
    column = frame.shape[1]
    ascii_art = ""
    for i in range(row):
        # Loop through each column of the current row
        for j in range(column):
            # print(i,j,frame[i][j])
            ascii_art += ASCII[frame[i][j]]  # Get corresponding ASCII character
        ascii_art += "\n"  # Add newline after each row

    print(ascii_art)


if __name__ == "__main__":
    # segment the human from background
    stdscr = curses.initscr()
    with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation:
        # read camera input
        camera = cv.VideoCapture(0)
        if not camera.isOpened():
            print("Cannot open camera")
            exit()
        while True:
            ret, frame = camera.read()
            if not ret:
                print("No frame received")
                break

            frame = cv.flip(frame,1) # flip horizontally

            ori_y, ori_x, _ = frame.shape
            max_y, max_x = stdscr.getmaxyx()

            # curses.start_color()
            # curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_WHITE)  # Black text on White background
            # stdscr.bkgd(' ', curses.color_pair(1))

            results = selfie_segmentation.process(frame)

            # convert to grayscale
            frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

            # frame = rembg.remove(frame) # not using as it is slow
            if results.segmentation_mask is not None:
                # The mask is a binary mask where the human region is white
                segmentation_mask = results.segmentation_mask
                segmentation_mask = np.where(segmentation_mask > 0.5, 255, 0).astype(np.uint8)
                
                frame = cv.bitwise_and(segmentation_mask,frame)

            # image quantization
            frame = np.round(frame/255*COUNT).astype(np.uint8)
            output_frame = (frame*(255/COUNT)).astype(np.uint8) # for testing visualization
            # print(output_frame.dtype)

            original_frame = output_frame.copy()

            frame = cv.resize(frame,(max_x-1,max_y-1),interpolation = cv.INTER_AREA)
            # frame = cv.resize(frame, (0,0), fx=0.2, fy=0.2, interpolation = cv.INTER_AREA)
            # max_x = os.get_terminal_size().columns-1
            # max_y = os.get_terminal_size().lines-1
            # frame = cv.resize(frame,(os.get_terminal_size().columns-1,os.get_terminal_size().lines-1),interpolation = cv.INTER_AREA)

            # if (max_x > max_y):
            #     ratio = ori_x/ori_y
            #     # frame = cv.resize(frame,(int((max_y-1)*ratio),max_y-1))
            #     frame = cv.resize(frame,(round(max_x//1.5),max_y-1))
            # else: # max_y > max_x
            #     ratio = ori_y/ori_x
            #     # frame = cv.resize(frame,(max_x-1,int((max_x-1)*ratio)))
            #     frame = cv.resize(frame,(max_x-1,round(max_y//1.5)))

            DisplayASCII(stdscr,frame)
            
            cv.imshow(WINDOWNAME,original_frame)
            cv.imshow(WINDOWNAME,original_frame)
            keyCode = cv.waitKey(1) # this is required for the video to show
            # if keyCode == ord('q'):
            #     break
            if cv.getWindowProperty(WINDOWNAME, cv.WND_PROP_VISIBLE) <1:
                break
    
    camera.release()
    cv.destroyAllWindows()
    curses.endwin()
    