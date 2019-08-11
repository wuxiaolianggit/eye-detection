import tensorflow as tf

def write_summary(summary_writer,tensor,summaryName,summaryType,step,max_images = 3):
    with summary_writer.as_default():
        if summaryType == 'scalar':
            tf.summary.scalar(summaryName,tensor,step=step)
        elif summaryType == 'image':
            tf.summary.image(summaryName,tensor,step=step,max_outputs=max_images)


def lr_decay_schedule(init_lr,epoch,decay_factor):
    return init_lr * (decay_factor ** epoch)

