

把VLM用Robotics Dataset Finetune，提升其VQA能力，同时又不损失其本身的QA能力。（冻结某些头可能可以实现，具体可以看相关论文）

我们可能需要同一个VLM Backbone，然后设计不同的projection来完成不同的任务。我们需要机器人视觉输入+指令->发掘因果关系，将因果关系应用到Robot动作指令生成（离散的？），如何用Flow Matching获得连续流畅的动作？

Now we are focusing on Causal Discovery Framework.