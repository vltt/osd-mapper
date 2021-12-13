**Algo:**
1) Sort files by hash.
Code: sorted_file_data
2) For each device, we compile a list of chunk intervals, the files from which are included in this
device.
Code: map_device_id_to_half_intervals
3) For each device, we consider how much memory we need to free (if the number is negative,
then this means how much memory we can fill, so as not to exceed the limit), write these numbers
into the list and sort them.
Code: need_free_space
4) We put two pointers - at the beginning and at the end of the list, that is, at the most empty and at
the most full. We follow the pointers to meet each other and call the function moving() that moves
the files.
5) Function moving(). It takes a pair of pointers l, r and returns which pointer needs to be shifted.
For each half-interval of the r-th device, in the deferred list of file hashes by bin search, we look for
the first file that is in this half-interval and starting from this file we go clockwise through the files
and for now we can add these files to the l-th device. There are three cases:
1. We reached the end of the half-interval, then instead of the old chunk we set a new one and
go to the next half-interval.
2. The current file is too large and cannot be added to the l-th device, then we exit the function
and return that we need to move the l pointer, that is, we will move it to the next device.
3. Already freed up enough space on the r-th device, then we exit the function and return that
we need to move the pointer r, then we will move the files from the next device.

**Why I chose this solution:**

It is logical to transfer files from the most full to the most empty. And specific files are not
particularly important to us, since the amount of them a lot more than amount of devices and each
individual file does not particularly affect the device’s fullness, so I can go in succession along a
half-interval and take files, so that I can move them with one chunk.
A plus of this approach: we create very few new chunks - maximum is number of devices, !not
chunks!, so this will not greatly affect the performance of the system as a whole, and we can use
this algorithm often without fear of quickly growing the number of chunks. That is, it is much better
to transfer many files at once to one chunk than to create many chunks.

**The maximum number of new chunks will be the number of devices.**

**Proof:**

Since we use two pointers, and we go maxim along the array until they meet, then we will go
through the array at most once.
For each position of 2 pointers pointing to devices (which move only towards each other) we call a
function that moves files to the need_free_space[l] device in a row from the prefix of the
half-interval of the need_free_space[r] device. In this function, there are three cases of adding new
chunks:
1) when we went through the entire half-interval, then we replace the chunk of the previous
device with a new one, therefore the number of instances does not increase
2) When we have already moved enough files and freed the cluster, we add one new chunk
and exit the function and reduce r by one.
3) When we cannot move anything, we add a new chunk, exit the function and increase l.
Thus, for each function call, we add a maximum of one new chunk.
That is, we will add the maximum number of device chunks.
