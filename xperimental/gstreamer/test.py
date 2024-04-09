"""
rtsp://rtsp:rtsp1234@oca142596.go.ro:556/Streaming/Channels/101

"""
import cv2
 


if __name__ == "__main__":
  info = cv2.getBuildInformation()
  print(info)
  
  rtsp_url = "rtsp://rtsp:rtsp1234@oca142596.go.ro:556/Streaming/Channels/601"
  gst_pipeline = f'rtspsrc location={rtsp_url} latency=0 ! rtph264depay ! avdec_h264 ! videoconvert ! appsink'
  
  cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
  print("Stream open {}".format("OK" if cap.isOpened() else "FAILED"))

  if cap.isOpened():
    done = False
    while cap is not None and not done:
      ret, frame = cap.read()
      if not ret or cv2.waitKey(1) & 0xFF == ord('q'):
        done = True
      else:
        # Process the frame
        cv2.imshow("Frame", frame)
      
    cap.release()
    cv2.destroyAllWindows()