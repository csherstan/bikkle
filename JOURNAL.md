2025-05-05

- Change env to produce variable numbers of each block type. I'm trying to figure out how the data is being handled
 in the replay buffer. When the data comes out of the replay buffer it is already in tensor format. Actually, do I
 really need to support this right now? I think I can just use the same number of each block type for now.
 I can always add this later.