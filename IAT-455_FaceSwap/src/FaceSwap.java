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

public class FaceSwap extends Frame {

	BufferedImage dectectedRect, detectCropped, featurePoints;

	public FaceSwap() {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

		Mat src = Imgcodecs.imread("src.jpg");

		Mat rect = FaceDetect.faceDetect(src, "rect");
		Mat cropped = FaceDetect.faceDetect(src, "cropped");
		Mat feature = FaceDetect.featurePoints(cropped);

		try {
			dectectedRect = convert(rect);
			detectCropped = convert(cropped);
			featurePoints = convert(feature);
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
		g.drawImage(featurePoints, 25 + dectectedRect.getWidth() / 2 + detectCropped.getWidth() + 100, 50, featurePoints.getWidth(),
				featurePoints.getHeight(), this);
	}

	public static void main(String[] args) {
		FaceSwap main = new FaceSwap();
		main.repaint();

	}

}
