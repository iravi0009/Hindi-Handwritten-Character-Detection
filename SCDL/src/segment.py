import cv2

def segment_characters(thresh_img):
    contours, _ = cv2.findContours(
        thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    chars = []
    for cnt in sorted(contours, key=lambda c: cv2.boundingRect(c)[0]):
        x,y,w,h = cv2.boundingRect(cnt)
        char_img = thresh_img[y:y+h, x:x+w]
        chars.append(char_img)
    return chars
