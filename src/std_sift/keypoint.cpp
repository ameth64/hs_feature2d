#include <vector>
#include <algorithm>
#include "hs_feature2d/std_sift/base_type.h"

namespace hs{
namespace feature2d{


struct KeyPoint_LessThan
{
    KeyPoint_LessThan(const std::vector<KeyPoint>& _kp) : kp(&_kp) {}
    bool operator()(int i, int j) const
    {
        const KeyPoint& kp1 = (*kp)[i];
        const KeyPoint& kp2 = (*kp)[j];
        if( kp1.pt.x != kp2.pt.x )
            return kp1.pt.x < kp2.pt.x;
        if( kp1.pt.y != kp2.pt.y )
            return kp1.pt.y < kp2.pt.y;
        if( kp1.size != kp2.size )
            return kp1.size > kp2.size;
        if( kp1.angle != kp2.angle )
            return kp1.angle < kp2.angle;
        if( kp1.response != kp2.response )
            return kp1.response > kp2.response;
        if( kp1.octave != kp2.octave )
            return kp1.octave > kp2.octave;
        if( kp1.class_id != kp2.class_id )
            return kp1.class_id > kp2.class_id;

        return i < j;
    }
    const std::vector<KeyPoint>* kp;
};

void KeyPointsFilter::removeDuplicated(std::vector<KeyPoint>& keypoints)
{
	int i, j, n = (int)keypoints.size();
	std::vector<int> kpidx(n);
	std::vector<Mat::Byte> mask(n, (Mat::Byte)1);

	for (i = 0; i < n; i++)
		kpidx[i] = i;
	std::sort(kpidx.begin(), kpidx.end(), KeyPoint_LessThan(keypoints));
	for (i = 1, j = 0; i < n; i++)
	{
		KeyPoint& kp1 = keypoints[kpidx[i]];
		KeyPoint& kp2 = keypoints[kpidx[j]];
		if (kp1.pt.x != kp2.pt.x || kp1.pt.y != kp2.pt.y ||
			kp1.size != kp2.size || kp1.angle != kp2.angle)
			j = i;
		else
			mask[kpidx[i]] = 0;
	}

	for (i = j = 0; i < n; i++)
	{
		if (mask[i])
		{
			if (i != j)
				keypoints[j] = keypoints[i];
			j++;
		}
	}
	keypoints.resize(j);
}


void KeyPointsFilter::retainBest(std::vector<KeyPoint>& keypoints, int n_points)
{
	//this is only necessary if the keypoints size is greater than the number of desired points.
	if (n_points >= 0 && keypoints.size() > (size_t)n_points)
	{
		if (n_points == 0)
		{
			keypoints.clear();
			return;
		}
		//first use nth element to partition the keypoints into the best and worst.
		std::nth_element(keypoints.begin(), keypoints.begin() + n_points, keypoints.end(), KeypointResponseGreater());
		//this is the boundary response, and in the case of FAST may be ambigous
		float ambiguous_response = keypoints[n_points - 1].response;
		//use std::partition to grab all of the keypoints with the boundary response.
		std::vector<KeyPoint>::const_iterator new_end =
			std::partition(keypoints.begin() + n_points, keypoints.end(),
			KeypointResponseGreaterThanThreshold(ambiguous_response));
		//resize the keypoints, given this new end point. nth_element and partition reordered the points inplace
		keypoints.resize(new_end - keypoints.begin());
	}
}







}
}