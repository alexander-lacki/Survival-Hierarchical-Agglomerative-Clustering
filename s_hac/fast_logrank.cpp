/*
Copyright (c) 2019, Andreas Stelzer
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of [project] nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// The original code was modified to accomodate different function inputs

bool compareFunction(std::pair<double, char>& a, std::pair<double, char>& b)
{
	return a.first < b.first;
}

std::vector<std::pair<double, char>> cpugetpairvector(std::vector<double>& vect1, std::vector<char>& vect2)
{
	std::vector<std::pair<double, char>> target(vect1.size());
	for (unsigned long long i = 0; i < target.size(); i++)
		target[i] = std::make_pair(vect1[i], vect2[i]);
	return target;
}

double logrank_instance(std::vector<double>& groupa, std::vector<char>& groupacensored, std::vector<double>& groupb, std::vector<char>& groupbcensored, bool teststatistic) {

	std::vector<std::pair<double, char>> groupaboth = cpugetpairvector(groupa, groupacensored);
	std::vector<std::pair<double, char>> groupbboth = cpugetpairvector(groupb, groupbcensored);

	std::sort(groupaboth.begin(), groupaboth.end(), compareFunction);
	std::sort(groupbboth.begin(), groupbboth.end(), compareFunction);

	unsigned long long na = groupaboth.size();
	unsigned long long nb = groupbboth.size();
	unsigned long long nasize = na;
	unsigned long long nbsize = nb;
	unsigned long long n = na + nb;
	long double oa = 0.0;
	long double ea = 0.0;
	long double var = 0.0;
	unsigned long long indexa = 0;
	unsigned long long indexb = 0;

	while (na > 0 && nb > 0) {
		double value = groupaboth[indexa].first;
		unsigned long long count = 0;
		unsigned long long countb = 0;
		unsigned long long minusna = 0;
		unsigned long long minusnb = 0;

		if (indexb < nbsize && groupbboth[indexb].first < value) {
			++minusnb;
			value = groupbboth[indexb].first;
			if (groupbboth[indexb].second == 1) {
				++countb;

			}
			++indexb;
			while (indexb < nbsize && value == groupbboth[indexb].first) {
				if (groupbboth[indexb].second == 1) {
					++countb;
				}
				++indexb;
				++minusnb;
			}
		}
		else if (indexb < nbsize && groupbboth[indexb].first == value) {
			++minusna;
			if (groupaboth[indexa].second == 1) {
				++count;

			}
			++indexa;
			while (indexa < nasize && value == groupaboth[indexa].first) {
				if (groupaboth[indexa].second == 1) {
					++count;
				}
				++indexa;
				++minusna;
			}
			while (indexb < nbsize && value == groupbboth[indexb].first) {
				if (groupbboth[indexb].second == 1) {
					++countb;
				}
				++indexb;
				++minusnb;
			}
		}
		else {
			++minusna;
			if (indexa < nasize && groupaboth[indexa].second == 1) {
				++count;

			}
			++indexa;
			while (indexa < nasize && value == groupaboth[indexa].first) {
				if (groupaboth[indexa].second == 1) {
					++count;
				}
				++indexa;
				++minusna;
			}
		}

		ea += static_cast<long double>(na * (count + countb)) / static_cast<long double>(n);
		oa += count;
		var += (static_cast<long double>(na) * static_cast<long double>(nb * (count + countb)) * static_cast<long double>(n - (count + countb))) / (static_cast<long double>(n) * static_cast<long double>(n) * static_cast<long double>(n - 1));
		na -= minusna;
		nb -= minusnb;
		n = na + nb;
	}
	long double logrank = ((oa - ea) * (oa - ea)) / var;
	if (teststatistic) {
		return static_cast<double>(logrank);
	}
	double pvalue = 1.0;
	if (logrank >= 0) {
		pvalue = (1.0 - boost::math::cdf(boost::math::chi_squared(1), logrank));
		//std::cout << pvalue << std::endl;
	}
	return pvalue;
}