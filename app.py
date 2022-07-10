import atexit
import platform
import queue
from apscheduler.scheduler import Scheduler
from flask import Flask,render_template,request
import threading
import requests
import os
import json
import matplotlib
import matplotlib.pyplot as plt
import io
import numpy as np
import time
from datetime import datetime
from PIL import Image
from torchvision import transforms
import cv2
import torch
import sys

global callRequests
callRequests=0
app = Flask(__name__)
cron = Scheduler()
cron.start()
lock=threading.Lock()

@app.route("/")
def home():
	return render_template("index.html")

@app.route("/videoload",methods=["POST","GET"])
def loading():
	def video_load():
		global callRequests
		if request.method=="POST":
			if request.files and callRequests < 3:
				try:
					callRequests+=1
					lock.acquire()
					video=request.files["video"]
					if (video.filename=="" or not(video.filename.lower().endswith(('.mp4','.webm','.avi','.mov','.mpeg','.mpg','.wmv','.m4v')))):
						callRequests-=1
						valueQueue.put(render_template("index.html"))
						return
					byte=video.read(10000000)
					sum=len(byte)
					while byte:
						byte=video.read(10000000)
						sum=sum+len(byte)
						if(sum>1000*1000*50):
							callRequests-=1
							valueQueue.put(render_template("index.html"))
							return;
					video.seek(0)
					del byte
					file=open("static/index.txt","r");
					integer=file.read()
					file.close()
					file=open("static/index.txt","w");
					file.write(str(int(integer)+1))
					file.close()
					html="<!DOCTYPE html> <html> <head> <title>Affective Computing</title> <meta charset=\"utf-8\"> <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"> <link rel=\"stylesheet\" href=\"https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/css/bootstrap.min.css\"> <link href=\"https://fonts.googleapis.com/css?family=Montserrat\" rel=\"stylesheet\"> <script src=\"https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js\"></script> <script src=\"https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/js/bootstrap.min.js\"></script> <link rel=\"stylesheet\" href=\"https://maxcdn.bootstrapcdn.com/bootstrap/4.5.0/css/bootstrap.min.css\"><script src=\"https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.16.0/umd/popper.min.js\"></script><script src=\"https://maxcdn.bootstrapcdn.com/bootstrap/4.5.0/js/bootstrap.min.js\"></script> <style> body {font: 20px Montserrat, sans-serif;line-height: 1.8;color: #f5f6f7;}p {font-size: 16px;}.margin {margin-bottom: 45px;}.bg-1 { background-color: grey;color: #ffffff;}.container-fluid {padding-top: 70px;padding-bottom: 70px;}.navbar {padding-top: 15px;padding-bottom: 15px;border: 0;border-radius: 0;margin-bottom: 0;font-size: 12px;letter-spacing: 5px;}.navbar-nav  li a:hover {color: #1abc9c !important;}</style></head> <body><nav class=\"navbar navbar-default\"><a href=\"static/downloadVideo"+integer+".mp4\" download><button type=\"button\" class=\"btn-lg btn-dark\">Download video</button></a><a href=\"static/logRecognition"+integer+".txt\" download><button type=\"button\" class=\"btn-lg btn-dark \">Download log</button></a><a href=\"/\"><button type=\"submit\" class=\"btn-lg btn-dark \">Home</button></a></nav> <div class=\"container-fluid bg-1 text-center\"><h1 class=\"margin\">Emotion Recognition ID: "+integer+"</h1> <video width=\"400\" height=\"400\" class=\"container-fluid bg-1 text-center\" controls=\"controls\">"
					plt.rcParams['figure.figsize'] = 10,5
					matplotlib.use('Agg')
					index=video.filename.find('.')
					path_video=os.path.join(os.getcwd(),(video.filename[:index]+integer+video.filename[index:]))
					video.save(path_video)
					mean=[0.0,0.0,0.0,0.0,0.0,0.0,0.0]
					numbers=[0,1,2,3,4,5,6]
					numbersInverted=[6,5,4,3,2,1,0]
					fig,(ax1,ax2)=plt.subplots(1,2)
					vid_capture = cv2.VideoCapture(path_video)
					vid_cod = cv2.VideoWriter_fourcc(*'H264')
					outputVideo = cv2.VideoWriter(os.path.join(os.getcwd(),"static/downloadVideo"+integer+".mp4"), vid_cod, 30, (1000,500))
					model = torch.load(os.path.join(os.getcwd(),'model_ft'),map_location=torch.device('cpu'))
					model.eval()
					num_frames=0.0
					file=open(os.path.join(os.getcwd(),"static/logRecognition"+integer+".txt"),"w")
					file.write("[["+"{\"Name\":\"downloadVideo"+integer+".mp4\""+", \"Size\":\""+str(sum)+"\"}],[")
					ret,frame = vid_capture.read()
					preprocess = transforms.Compose([
						transforms.Resize(256),
						transforms.CenterCrop(224),
						transforms.ToTensor(),
						transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
					])
					emotions=("Angry", "Disgust", "Fear", "Happy", "Neutral","Sad","Surprise")
					setattr(request,"video","static/downloadVideo"+integer+".mp4")
					lock.release()
					while(ret):
						lock.acquire()
						ax1=plt.subplot(1,2,1)
						ax2=plt.subplot(1,2,2)
						before=time.time()
						cv2_im=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
						pil_im = Image.fromarray(cv2_im)
						num_frames=num_frames+1
						input_tensor=preprocess(pil_im)
						input_batch=input_tensor.unsqueeze(0)
						with torch.no_grad():
							output=model(input_batch)
							tensor=torch.softmax(output,dim=1)
							values,index=tensor.topk(7)
						ax1.imshow(cv2_im)
						array=[]
						array2=[]
						array_color=[]
						ax2.set_xlim(0.01,1)
						array_index=index.numpy()[0]
						for x in numbers:
							if(array_index[0]==x):
								mean[x]=mean[x]+1
						for x in array_index:
							array.append(emotions[x])
							if(x==0):
								array_color.append('red')
							if(x==1):
								array_color.append('brown')
							if(x==2): 
								array_color.append('black')
							if(x==3): 
								array_color.append('yellow')
							if(x==4): 
								array_color.append('gray')
							if(x==5): 
								array_color.append('blue')
							if(x==6): 
								array_color.append('orange')
						array.reverse()
						array_color.reverse()
						array_values=values.numpy()[0]
						for x in array_values:
							array2.append(x)
						array2.reverse()
						ax2.barh(array,array2,align='center',color=array_color)
						ax2.set_yticks(array)
						ax2.set_yticklabels(array) 
						ax2.set_xlabel('Probability')
						ax2.set_title('Emotion Recognition')
						ax1.set_title(array[6])
						file.write("{\"Frame\":\""+str(num_frames)+"\",")
						for x in numbersInverted:
							if (x>0):
								file.write("\""+str(array[x])+"\":"+"\""+str(array2[x])+"\",")
							else:
								file.write("\""+str(array[x])+"\":"+"\""+str(array2[x])+"\"},")
						buf=io.BytesIO()
						plt.savefig(buf,format='jpg')
						buf.seek(0)
						plt.clf()
						img_array=np.asarray(bytearray(buf.read()),dtype=np.uint8)
						img=cv2.imdecode(img_array,1)
						outputVideo.write(img)
						ret,frame = vid_capture.read()
						print("Loading frame n° "+str(num_frames)+" Time: "+str(time.time()-before))
						lock.release()
					html=html+" <source src=\""+getattr(request,"video")+"\" type=\"video/mp4\"/></video></div>"
					vid_capture.release()
					cv2.destroyAllWindows()
					file.write("{\"TotalFrames\":"+str(num_frames)+"}],[")
					for x in numbers:
						if(mean[x]>=0):
							if (x<6):
								file.write("{\"Emotion\":\""+emotions[x]+"\",\"Average probability\":\""+str(mean[x]/num_frames*100)+"\"}"+",")
							else:
								file.write("{\"Emotion\":\""+emotions[x]+"\",\"Average probability\":\""+str(mean[x]/num_frames*100)+"\"}")
					file.write("]]\n")
					file.close()
					os.remove(path_video)
					valueQueue.put(html+"</body></html>")
					callRequests-=1
				except Exception as err:
					callRequests-=1
					print("catched Exception: "+err)
					valueQueue.put(render_template("index.html"))
					return
			else:
				valueQueue.put(render_template("index.html"))
				return
		else:
			valueQueue.put(render_template("index.html"))
			return
	
	valueQueue = queue.Queue()
	thread=threading.Thread(target=video_load())
	thread.start()
	thread.join()
	return valueQueue.get()


@app.route("/api/log/<int:file_id>",methods=["GET"])
def get_log(file_id):
	for file in os.listdir(os.getcwd()+"\\static"):
		if file.endswith(str(file_id)+".txt"):
			file=open("static/logRecognition"+str(file_id)+".txt","r")
			writer=file.read()
			file.close()
			return "<!DOCTYPE html><html><body>"+writer+"</body></html>" 
	return render_template("index.html")

def job_function():
	for file in os.listdir(os.getcwd()+"/static"):
		if not(file.endswith("affective.jpg") or file.endswith(".txt")): 
			if platform.system() == 'Windows':
				stat = os.path.getctime(os.path.join(os.getcwd()+"/static",file))
			else:
				stat = os.stat.st_mtime(os.path.join(os.getcwd()+"/static",file))

			if time.time()-stat>10800:
				os.remove(os.path.join(os.getcwd()+"/static",file))
				print(os.path.join(os.getcwd()+"/static",file)+" removed")

	for file in os.listdir(os.getcwd()):
		if not(file.endswith("static") or file.endswith("app.py") or file.endswith("templates") or file.endswith("ft")): 
			if platform.system() == 'Windows':
				stat = os.path.getctime(os.path.join(os.getcwd(),file))
			else:
				stat = os.stat.st_mtime(os.path.join(os.getcwd(),file))

			if time.time()-stat>3600:
				os.remove(os.path.join(os.getcwd(),file))
				print(os.path.join(os.getcwd(),file)+" removed")

if __name__ == "__main__":
	app.run(debug=True)
	cron.add_cron_job(job_function, hour='0')
