# Saving model with best validation F1-Score
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_f1_c', verbose=1, save_best_only=True, mode='max', period=1)
tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)

# Callback to save all models weights
class SaveWeightsCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    print("Saving weights for epoch {}".format(epoch))
    model.save_weights('hms/weights_epoch{}.h5'.format(epoch))  
class PlotLearning(keras.callbacks.Callback):
  """
  Plotting the F1-Score and Loss over the training
  """
  def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        
        self.fig = plt.figure()
        
        self.logs = []

  def on_train_batch_end(self, epoch, logs={}):
        
        if(epoch % 10 == 0):
            self.logs.append(logs)
            self.x.append(self.i)
            self.losses.append(logs.get('loss'))
            self.val_losses.append(logs.get('f1_c'))
            self.i += 1

            clear_output(wait=True)


            fig, ax1 = plt.subplots()

            color_r = 'tab:red'
            ax1.set_xlabel('time (s)')
            ax1.set_ylabel('loss', color=color_r)
            ax1.plot(self.x, self.losses, color=color_r)
            ax1.tick_params(axis='y', labelcolor=color_r)

            ax2 = ax1.twinx()

            color = 'tab:blue'
            ax2.set_ylabel('f1_c', color=color)  # we already handled the x-label with ax1
            ax2.plot(self.x, self.val_losses)
            ax2.tick_params(axis='y')

            plt.legend()
            plt.show()
plot = PlotLearning()
plot   