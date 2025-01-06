
import cv2 as cv
import numpy as np
import mediapipe as mp
import curses
import os
# import rembg # not using as it is slow


WINDOWNAME = "camera"
# https://paulbourke.net/dataformats/asciiart/
ASCII  = "@%#*+=-:." 
COUNT = len(ASCII)-1
COLOR_COUNT = 3


mp_selfie_segmentation = mp.solutions.selfie_segmentation
mp_drawing = mp.solutions.drawing_utils

def DisplayASCII(stdscr, frame):
    stdscr.clear()  # Clear the screen
    row = frame.shape[0]
    column = frame.shape[1]

    # Create a list to hold the ASCII art lines
    ascii_art = ""

    for i in range(row):
        # Loop through each column of the current row
        for j in range(column):
            frame_value = frame[i][j]
            color_value = round(frame_value/COUNT*COLOR_COUNT)
            stdscr.attron(curses.color_pair(color_value))
            stdscr.addstr(ASCII[frame_value])
            stdscr.attroff(curses.color_pair(color_value))
            # ascii_art += ASCII[frame_value]  # Get corresponding ASCII character
        ascii_art += "\n"  # Add newline after each row
        stdscr.addstr("\n")

    # Add the entire ASCII art at once to the screen
    # stdscr.addstr(ascii_art
    stdscr.refresh()  # Refresh the screen to display the updated content

if __name__ == "__main__":
    # segment the human from background
    stdscr = curses.initscr()
    stdscr.nodelay(True)
    # read camera input
    camera = cv.VideoCapture(0)
    if not camera.isOpened():
        print("Cannot open camera")
        exit()

    # https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/selfie_segmentation.md#models
    selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
    while True:
        ret, frame = camera.read()
        if not ret:
            print("No frame received")
            break

        frame = cv.flip(frame,1) # flip horizontally
        ori_y, ori_x, _ = frame.shape
        max_y, max_x = stdscr.getmaxyx()

        curses.start_color()
        bg_color = curses.COLOR_WHITE
        fg_color = curses.COLOR_BLACK
        color1 = curses.COLOR_BLUE
        color2 = curses.COLOR_GREEN
        color3 = curses.COLOR_RED

        # bg_color = curses.COLOR_BLACK
        # fg_color = curses.COLOR_WHITE
        # color1 = curses.COLOR_GREEN
        # color2 = curses.COLOR_CYAN
        # color3 = curses.COLOR_YELLOW

        curses.init_pair(1, color1, bg_color)
        curses.init_pair(2, color2, bg_color)
        curses.init_pair(3, color3, bg_color)
        curses.init_pair(4, fg_color, bg_color) # default background
        stdscr.bkgd(" ",curses.color_pair(4))

        
        dilate_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))
        results = selfie_segmentation.process(frame)
        # frame = rembg.remove(frame) # not using as it is slow
        if results.segmentation_mask is not None:
            # The mask is a binary mask where the human region is white
            segmentation_mask = results.segmentation_mask
            segmentation_mask = np.where(segmentation_mask > 0.5, 255, 0).astype(np.uint8)
            segmentation_mask = cv.dilate(segmentation_mask, dilate_kernel) # make the foreground show more
            segmentation_mask = np.stack([segmentation_mask]*3,axis=-1) # transform it to 3 channel
            frame = cv.bitwise_and(segmentation_mask,frame) # bitwise AND the 3 channel

        frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY) # convert to grayscale
        frame = cv.GaussianBlur(frame,(3,3),3) # remove noises
        frame = cv.equalizeHist(frame)  # histogram equalization

        original_frame = cv.Canny(frame,100,200)
        original_frame = cv.dilate(original_frame,dilate_kernel)
        original_frame = cv.bitwise_and(original_frame,frame)
        # original_frame = frame.copy()

        # image quantization
        frame = np.round(frame/255*COUNT).astype(np.uint8)
        output_frame = np.round(frame*(255/COUNT)).astype(np.uint8) # for testing visualization
        # print(output_frame.dtype)

        if (max_x > max_y):
            ratio = ori_x/ori_y
            new_x = round((max_y-1)*ratio*2) # *2 the size to make it look better 
            new_x = max_x if new_x > max_x else new_x
            new_y = max_y-1
            frame = cv.resize(frame,(new_x,new_y))
            
        else: # max_y > max_x
            ratio = ori_y/ori_x
            new_x = max_x-1
            new_y = round((max_x-1)*ratio*2) # *2 the size to make it look better 
            new_y = max_y if new_y > max_y else new_y
            frame = cv.resize(frame,(new_x,new_y))
            

        DisplayASCII(stdscr,frame)
        key_pressed = stdscr.getch()
        if (key_pressed == ord("q") or key_pressed == 27) : # ESC ASCII value is 27
            break
        
        # This is for opencv to show the camera input
        # cv.imshow(WINDOWNAME,original_frame)
        # keyCode = cv.waitKey(1) # this is required for the video to show
        # if keyCode == ord('q'):
        #     break
        # if cv.getWindowProperty(WINDOWNAME, cv.WND_PROP_VISIBLE) <1:
        #     break
    
    camera.release()
    cv.destroyAllWindows()
    curses.endwin()
    