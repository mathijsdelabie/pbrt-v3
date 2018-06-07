/*
 * ReverseMapping.cpp
 *
 *  Created on: 21 Nov 2017
 *      Author: Mathijs
 */

#include "reversemapping.h"
#include "sampling.h"

namespace pbrt {

	//STAT_PERCENT("Integrator/Crossover visibility", invisibleInversions, totalInversions);

	ReverseMapping::ReverseMapping() = default;

	ReverseMapping::~ReverseMapping() = default;

	void ReverseMapping::ReverseMap(GMLTSampler &sampler){
		int t = sampler.GetCurrentT();
		int s = sampler.GetCurrentS();
		std::vector<Vertex> &cv = sampler.GetCurrentCV();
		std::vector<Vertex> &lv = sampler.GetCurrentLV();

		int n = sampler.GetLength();
		for (int i = 0; i < n; ++i) {
			int newI;
			int stream = sampler.GetStreamForIndex(i, newI);

			if(stream == 2 || (stream == 0 && (sampler.GetDepth() == 0 || i<6)) || (stream == 1 && i<5)){
				 sampler.SetXi(stream, newI, sampler.GetXi(stream, newI));
			}
		}

		for (int i = 2; i < t; ++i) {
			BSDF bsdf = BSDF(cv[i-1].si);
			Vector3f wi = bsdf.WorldToLocal(cv[i].si.p - cv[i-1].si.p);
			//Vector3f wi = bsdf.WorldToLocal(cv[i].si.wo*-1);
			Point2f u = InverseConcentricSampleDisk(wi);
			Point2f v( sampler.GetXi(0, 2 + 2*i) , sampler.GetXi(0, 3 + 2*i));
			//assert(std::abs(u.x-v.x) < 0.0000001);
			//assert(std::abs(u.y-v.y) < 0.0000001);
			assert(u.x >= 0 && u.x <= 1);
			assert(u.y >= 0 && u.y <= 1);

			sampler.SetXi(0, 2 + 2*i, u.x);
			sampler.SetXi(0, 3 + 2*i, u.y);
		}

		for (int i = 2; i < s; ++i) {
			BSDF bsdf = BSDF(lv[i-1].si);
			Vector3f wi = bsdf.WorldToLocal(lv[i].si.p - lv[i-1].si.p);
			//Vector3f wi = bsdf.WorldToLocal(lv[i].si.wo*-1);
			Point2f u = InverseConcentricSampleDisk(wi);
			Point2f v( sampler.GetXi(1, 1 + 2*i) , sampler.GetXi(1, 2 + 2*i));
			//assert(std::abs(u.x-v.x) < 0.0000001);
			//assert(std::abs(u.y-v.y) < 0.0000001);
			assert(u.x >= 0 && u.x <= 1);
			assert(u.y >= 0 && u.y <= 1);

			sampler.SetXi(1, 1 + 2*i, u.x);
			sampler.SetXi(1, 2 + 2*i, u.y);
		}
	}

	Point2f ReverseMapping::InverseConcentricSampleDisk(Vector3f &wi){
		//if(wi.z > 0) ++invisibleInversions;
		//++totalInversions;
		if(wi.x == 0 && wi.y == 0) return Point2f(0.5,0.5);

		Vector3f v = Normalize(wi);

		Float r = std::min(1., std::sqrt(v.x*v.x + v.y*v.y));
		Float thetaDivPiOver4 = std::atan2(v.y,v.x)/PiOver4;

		if(thetaDivPiOver4 > 3 || thetaDivPiOver4 <= -1){
			r *= -1;
			thetaDivPiOver4 = std::fmod(thetaDivPiOver4+8, 8) - 4;
		}

		Point2f u;
		if(thetaDivPiOver4 < 1){
			u.x = r;
			u.y = r * thetaDivPiOver4;
		}
		else{
			u.y = r;
			u.x = -r * (thetaDivPiOver4 - 2);
		}

		return (u + Vector2f(1, 1))*0.5;
	}


}

