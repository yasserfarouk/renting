FILE_TO_PATCH=`python -c "from ray.rllib.utils.spaces import space_utils; print(space_utils.__file__)"`
echo Attempting to patch $FILE_TO_PATCH
patch -f $FILE_TO_PATCH < patches/space_utils.patch
