import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.features2d.Features2d;
import org.opencv.features2d.ORB;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

public class FaceDetect {

	public static Mat faceDetect(Mat src, String resultType) {
		CascadeClassifier sample = new CascadeClassifier(
				"E:\\opencv\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_alt_tree.xml");
		MatOfRect result = new MatOfRect();
		Mat croppedImage = new Mat();
		sample.detectMultiScale(src, result);

		if (result.toArray().length < 1) {
			return src;
		}

		for (Rect rect : result.toArray()) {
			Imgproc.rectangle(src, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height),
					new Scalar(255, 255, 255), 1);

			croppedImage = new Mat(src,
					new Rect(new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height)));
		}

		if (resultType == "rect") {
			return src;
		} else if (resultType == "cropped") {
			return croppedImage;
		}
		return null;
	}

	public static Mat eyeDetect(Mat src) {
		CascadeClassifier sample = new CascadeClassifier(
				"E:\\opencv\\opencv\\sources\\data\\haarcascades\\haarcascade_eye_tree_eyeglasses.xml");

		MatOfRect result = new MatOfRect();
		sample.detectMultiScale(src, result);

		if (result.toArray().length < 1) {
			return src;
		}

		for (Rect rect : result.toArray()) {
			Imgproc.rectangle(src, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height),
					new Scalar(255, 255, 255), 1);
		}
		return src;
	}
	
	public static Mat featurePoints (Mat mat) {
		Mat result = new Mat();
		Mat gray = new Mat();
		Imgproc.cvtColor(mat, gray, Imgproc.COLOR_BGR2GRAY);
		ORB orb = ORB.create(50,2,8,10,0,2,ORB.HARRIS_SCORE,10,20);
		MatOfKeyPoint pt = new MatOfKeyPoint();
		orb.detect(gray, pt);
		
		Features2d.drawKeypoints(mat, pt, result, new Scalar(255,255,255),Features2d.DrawMatchesFlags_DRAW_RICH_KEYPOINTS);
		return result;
	}

}
