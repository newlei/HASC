Original image id(real image id in flicker): img_id_before_3w.txt
we reid the real image id. (e.g.,real image id in the first line->reid:0, real image id in the second line ->reid:1)

Original user id(real user id in flicker): user_id_before_3w.txt
we reid the real user id. (e.g.,real user id in the first line->reid:0, real user id in the second line ->reid:1)

user_favor.txt
user favors list of image. 
each line: user-id favored-image-id favored-image-id, ... favored-image-id.

user_follow.txt
user follows list of user. 
each line: user-id followed-user-id followed-user-id, ... followed-user-id.

user_up.txt
user uploads the list of image. 
each line: user-id followed-user-id followed-user-id, ... followed-user-id.

For convenience, we transpose user_up.txt to up_user.txt

