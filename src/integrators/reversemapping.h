/*
 * ReverseMapping.h
 *
 *  Created on: 21 Nov 2017
 *      Author: Mathijs
 */

#ifndef SRC_INTEGRATORS_REVERSEMAPPING_H_
#define SRC_INTEGRATORS_REVERSEMAPPING_H_

#include "integrators/gmlt.h"
#include "integrators/bdpt.h"

namespace pbrt {

	class ReverseMapping {
		public:
			ReverseMapping();
			virtual ~ReverseMapping();

			void ReverseMap(GMLTSampler &sampler);
		private:
			Point2f InverseConcentricSampleDisk(Vector3f &wi);
	};

}

#endif /* SRC_INTEGRATORS_REVERSEMAPPING_H_ */
