from sklearn.externals import joblib
import dataset
import argparse
import mahotas
import cv2
import mnist_classify


image = cv2.imread("5.jpeg")
cv2.imshow("4", image)
cv2.waitKey(0)


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print(gray)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 30, 150)

cv2.imshow("edged", edged)
cv2.waitKey(0)
_, contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contours = sorted([(c, cv2.boundingRect(c)[0]) for c in contours], key=lambda x: x[1])

for (c, _) in contours:
    (x, y, w, h) = cv2.boundingRect(c)

    if w >= 7 and h > 20:
        roi = gray[y:y + h, x:x + w]
        thresh = roi.copy()
        T = mahotas.thresholding.otsu(roi)
        thresh = cv2.bitwise_not(thresh)

        thresh = dataset.deskew(thresh, 28)
        thresh = dataset.center_extent(thresh, (28, 28))

        cv2.imshow("thresh", thresh)
        print(thresh)
        cv2.waitKey(0)

        amin, amax = thresh.min(), thresh.max()  # 求最大最小值
        a = (thresh - amin) / (amax - amin)  # (矩阵元素-最小值)/(最大值-最小值)
        print(a)
        #hist = hog.describe(thresh)
        #digit = model.predict([hist])[0]
        digit = mnist_classify.predict(a)
        print("I think that number is: {}".format(digit))

        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0))
        cv2.putText(image, str(digit), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        cv2.imshow("image", image)
        cv2.waitKey(0)