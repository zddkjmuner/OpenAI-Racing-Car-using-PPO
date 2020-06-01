# OpenAI-Racing-Car-using-PPO
This's a final project for UPenn CIS 680 (Vision and Learning). We trained the agent using PPO.


We train the model in Google Colab. You can choose to train with visualization by setting “self.render = True” in Arguments class. While in colab, visualization is not supported, so we choose to train on the colab using GPU and test on the laptop CPU. The gym version is v0.15.4. If you want to train using your laptop CPU, and if your version is higher than v0.9.5, there might be some problems with visualizing. 

To train the model, you can just execute the code part block by block till the “Train” part. The model is saved in the directory “My Drive/CIS680_2019/FinalProject” named as “ppo_params.pth”.
To test the model, we provided one that have already been trained, or you can also use your own model and put the file in the “param” folder. In the terminal, under the project path, execute “python test.py”, and the testing will start. If you want to visualize the testing process, use the line “python test.py --render”.

We also included a 6-min video presenting our result in the “try_catch_me.ppt” file. More details about our approaches and results are included in the report file.
