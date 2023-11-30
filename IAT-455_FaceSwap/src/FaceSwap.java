
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
import org.opencv.photo.Photo;

public class FaceSwap extends Frame {

	BufferedImage detectCropped, test;

	public FaceSwap() {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

		Mat src = Imgcodecs.imread("src1.jpg");
		Mat target = Imgcodecs.imread("1.jpg");

		Mat cropped1 = FaceDetect.faceDetect(src);
		Mat cropped2 = FaceDetect.faceDetect(target);

		List<MatOfPoint> hull1 = hullpoints(cropped1);
		List<MatOfPoint> hull2 = hullpoints(cropped2);

		List<Point> hullpoints1 = new ArrayList<>();
		List<Point> hullpoints2 = new ArrayList<>();

		for (MatOfPoint matOfPoint : hull1) {
			hullpoints1.addAll(matOfPoint.toList());
		}
		for (MatOfPoint matOfPoint : hull2) {
			hullpoints2.addAll(matOfPoint.toList());
		}

		Size newSize = new Size(cropped2.width(), cropped2.height());
		Imgproc.resize(cropped1, cropped1, newSize);
		Mat test2 = triangle(cropped1, cropped2, hullpoints2);

		//
		
		Rect rect2 = FaceDetect.rect(target);
		test2.copyTo(new Mat(target, rect2));

		Mat test1 = target.clone();
		//

		try {
			detectCropped = convert(cropped1);
			test = convert(test1);

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

		g.drawImage(test, 75, 50, test.getWidth(), test.getHeight(), this);
	}

	public static void main(String[] args) {
		FaceSwap main = new FaceSwap();
		main.repaint();

	}

	// ConvexHull
	// https://docs.opencv.org/4.x/d7/d1d/tutorial_hull.html
	public List<MatOfPoint> hullpoints(Mat src) {
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
		return hullList;

	}

	// test
	public Mat triangle(Mat src, Mat target, List<Point> points) {

		List<Point> hull = new ArrayList<>();
		for (int i = 0; i < points.size(); i++) {
			Point pt = new Point((int) points.get(i).x, (int) points.get(i).y);
			hull.add(pt);
		}
		Mat mask = Mat.zeros(target.rows(), target.cols(), target.type());
		Imgproc.fillConvexPoly(mask, new MatOfPoint(hull.toArray(new Point[0])), new Scalar(255, 255, 255));

		Rect r = Imgproc.boundingRect(new MatOfPoint2f(hull.toArray(new Point[0])));
		Point center = new Point((r.tl().x + r.br().x) / 2, (r.tl().y + r.br().y) / 2);

		src.convertTo(src, CvType.CV_8UC3);

		Mat output = new Mat();

		Photo.seamlessClone(src, target, mask, center, output, Photo.NORMAL_CLONE);

		return output;

	}

}
