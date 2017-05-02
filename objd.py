import cv2

finger_haar = cv2.CascadeClassifier("cascade.xml")
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
while(ret):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    finger = finger_haar.detectMultiScale(gray, 2.2, 5)
    for finger_X, finger_y, finger_w, finger_h in finger:
        cv2.rectangle(frame, (finger_X, finger_y), (finger_X+112, finger_y+112), (0, 255, 0), 2)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    ret, frame = cap.read()

cap.release()
# cv2.destroyAllWindows()
