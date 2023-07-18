#
import cv2

class Cv2AviMp4:
    def __init__(self):
        self.name = 'scripts.cv2_avi_mp4.Cv2AviMp4'

    @staticmethod
    def convert_avi_mp4(avi_file: str, mp4_file: str):
        #获得视频的格式  
        videoCapture = cv2.VideoCapture(avi_file)  
        #获得码率及尺寸  
        fps = videoCapture.get(cv2.CAP_PROP_FPS)  
        size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),   
                    int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))  
        #指定写视频的格式, I420-avi, MJPG-mp4  
        # videoWriter = cv2.VideoWriter(mp4_file, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, size)  
        videoWriter = cv2.VideoWriter(mp4_file, cv2.VideoWriter_fourcc('M', 'P', '4', 'V'), fps, size)  
        #读帧  
        success, frame = videoCapture.read()
        frameNum = 1  
        while success :  
            # cv2.imshow(ad3 Video, frame) #显示  
            cv2.waitKey(int(1000/fps)) #延迟  
            videoWriter.write(frame) #写视频帧  
            success, frame = videoCapture.read() #获取下一帧 
            print(f'processed {frameNum} frame')
            frameNum += 1

if '__main__' == __name__:
    avi_files = [
        './work/test0_normal.avi',  './work/test1_crowd.avi',  './work/test2_hlight.avi',  
        './work/test3_shadow.avi',  './work/test4_noline.avi',  './work/test5_arrow.avi',  
        './work/test6_curve.avi',  './work/test7_cross.avi'  './work/test8_night.avi'
    ]
    for avi_file in avi_files:
        mp4_file = f'{avi_file[:-4]}.mp4'
        print(f'convert {avi_file} => {mp4_file} ...')
        Cv2AviMp4.convert_avi_mp4(avi_file, mp4_file)



   

   

   

   
