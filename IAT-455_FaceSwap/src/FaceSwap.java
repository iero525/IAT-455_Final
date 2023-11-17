import java.awt.Frame;
import java.awt.Graphics;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import javax.imageio.ImageIO;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

public class FaceSwap extends Frame {

	BufferedImage dectectedRect, detectCropped, featurePoints, hullImg;

	public FaceSwap() {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

		Mat src = Imgcodecs.imread("src.jpg");

		Mat rect = FaceDetect.faceDetect(src, "rect");
		Mat cropped = FaceDetect.faceDetect(src, "cropped");
		Mat feature = FaceDetect.featurePoints(cropped);
		Mat hull = hull(cropped);

		try {
			dectectedRect = convert(rect);
			detectCropped = convert(cropped);
			featurePoints = convert(feature);
			hullImg = convert(hull);

		} catch (IOException e) {
			System.out.println("Convert Error");
		}

		this.setTitle("IAT-455 FACE SWAP");
		this.setVisible(true);

		this.addWindowListener(new WindowAdapter() {
			public void windowClosing(WindowEvent e) {
				System.exit(0);
			}
		});
	}

	// https://www.tutorialspoint.com/how-to-convert-opencv-mat-object-to-bufferedimage-object-using-java
	public static BufferedImage convert(Mat mat) throws IOException {
		// Encoding the image
		MatOfByte matOfByte = new MatOfByte();
		Imgcodecs.imencode(".jpg", mat, matOfByte);
		// Storing the encoded Mat in a byte array
		byte[] byteArray = matOfByte.toArray();
		// Preparing the Buffered Image
		ByteArrayInputStream in = new ByteArrayInputStream(byteArray);
		BufferedImage bufImage = ImageIO.read(in);
		return bufImage;
	}

	public void paint(Graphics g) {

		this.setSize(1500, 800);

		g.drawImage(dectectedRect, 50, 50, dectectedRect.getWidth() / 2, dectectedRect.getHeight() / 2, this);
		g.drawImage(detectCropped, 25 + dectectedRect.getWidth() / 2 + 50, 50, detectCropped.getWidth(),
				detectCropped.getHeight(), this);
		g.drawImage(featurePoints, 50 + dectectedRect.getWidth() / 2 + detectCropped.getWidth() + 50, 50,
				featurePoints.getWidth(), featurePoints.getHeight(), this);
		g.drawImage(hullImg,
				75 + dectectedRect.getWidth() / 2 + detectCropped.getWidth() + featurePoints.getWidth() + 50, 50,
				hullImg.getWidth(), hullImg.getHeight(), this);
	}

	public static void main(String[] args) {
		FaceSwap main = new FaceSwap();
		main.repaint();

	}
	
	//TESTING
	// https://docs.opencv.org/4.x/d7/d1d/tutorial_hull.html
	public Mat hull(Mat src) {
		Mat srcGray = new Mat();
		int threshold = 100;

		Imgproc.cvtColor(src, srcGray, Imgproc.COLOR_BGR2GRAY);
		Imgproc.blur(srcGray, srcGray, new Size(3, 3));

		Mat cannyOutput = new Mat();
		Imgproc.Canny(srcGray, cannyOutput, threshold, threshold * 2);

		List<MatOfPoint> contours = new ArrayList<>();
		Mat hierarchy = new Mat();
		Imgproc.findContours(cannyOutput, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);

		List<MatOfPoint> hullList = new ArrayList<>();
		for (MatOfPoint contour : contours) {
			MatOfInt hull = new MatOfInt();
			Imgproc.convexHull(contour, hull);
			Point[] contourArray = contour.toArray();
			Point[] hullPoints = new Point[hull.rows()];
			List<Integer> hullContourIdxList = hull.toList();
			for (int i = 0; i < hullContourIdxList.size(); i++) {
				hullPoints[i] = contourArray[hullContourIdxList.get(i)];
			}
			hullList.add(new MatOfPoint(hullPoints));
		}

		Mat drawing = Mat.zeros(cannyOutput.size(), CvType.CV_8UC3);
		for (int i = 0; i < contours.size(); i++) {
			Scalar color = new Scalar(255, 255, 255);
//			Imgproc.drawContours(drawing, contours, i, color);
			Imgproc.drawContours(drawing, hullList, i, color);
		}
		return drawing;
	}
	
	
	
}
