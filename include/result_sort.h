#ifndef _RESULT_SORT_H
#define _RESULT_SORT_H

#include <iostream>
#include <vector>
#include <utility>
#include <algorithm>

struct Comp{
	Comp( const std::vector<float>& v ) : _v(v) {}
	bool operator ()(int a, int b) { return _v[a] > _v[b]; }
	const std::vector<float>& _v;
};

static void result_sort(std::vector<float> & ary,int topN ,std::vector<std::pair<int,float> > & topResult)
{
	int cnt= ary.size(); //取一次值 保持效率

	if(topN> cnt ) 
		topN= cnt;

	std::vector<int> temp;
	temp.resize(cnt);
	for ( int i= 0; i< cnt; ++i ) 
		temp[i]= i;

	partial_sort(temp.begin(), temp.begin()+topN, temp.end(), Comp(ary));


	for (int i=0; i<topN; i++)
	{
		int index = temp[i];
        topResult.push_back(std::make_pair(index,ary[index]));
		//cout << "\t" << index << "th: " << ary[index] << "\n";
	}
}
#endif
