
/*
    pbrt source code is Copyright(c) 1998-2016
                        Matt Pharr, Greg Humphreys, and Wenzel Jakob.

    This file is part of pbrt.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
    TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 */

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_INTEGRATORS_GMLT_H
#define PBRT_INTEGRATORS_GMLT_H

// integrators/gmlt.h*
#include "pbrt.h"
#include "integrator.h"
#include "integrators/bdpt.h"
#include "sampler.h"
#include "spectrum.h"
#include "film.h"
#include "rng.h"
#include <unordered_map>

namespace pbrt {

// GMLTSampler Declarations
class GMLTSampler : public Sampler {
  public:
    // GMLTSampler Public Methods
    GMLTSampler(int mutationsPerPixel, int rngSequenceIndex, Float sigma,
               Float largeStepProbability, int streamCount)
        : Sampler(mutationsPerPixel),
          rng(rngSequenceIndex),
          sigma(sigma),
          largeStepProbability(largeStepProbability),
          streamCount(streamCount) {}
    GMLTSampler(const GMLTSampler &) = default;
    GMLTSampler(GMLTSampler &&) = default;
    Float Get1D();
    Point2f Get2D();
    std::unique_ptr<Sampler> Clone(int seed);
    void StartIteration();
    bool IsNoneZeroIteration(){ return currentIteration > 0; };
    void Accept();
    void Reject();
    void StartStream(int index);
    int GetNextIndex() { return sampleIndex++; }

    int GetLength() { return camX.size() + lightX.size() + conX.size(); }
    int GetStreamForIndex(int index, int &newIndex);
    Float GetXi(int stream, int index);
    void SetXi(int stream, int index, Float value);
    void SetCrossover(bool doCrossover) { crossover = doCrossover; }

    void SetProposedT(int t){ proposed.t = t; }
    void SetProposedS(int s){ proposed.s = s; }
    Vertex* GetProposedCV(int t){
    	proposed.cv.resize(t);
    	return proposed.cv.data();
    }
    Vertex* GetProposedLV(int s){
    	proposed.lv.resize(s);
        return proposed.lv.data();
    }
    int GetCurrentT() { return current.t; };
    int GetCurrentS() { return current.s; };
    std::vector<Vertex> &GetCurrentCV() { return current.cv; };
    std::vector<Vertex> &GetCurrentLV() { return current.lv; };
    int GetDepth() { return current.t+current.s-2; }

  protected:
    // GMLTSampler Private Declarations
    struct PrimarySample {
        Float value = 0;
        // PrimarySample Public Methods
        void Backup() {
            valueBackup = value;
            modifyBackup = lastModificationIteration;
        }
        void Restore() {
            value = valueBackup;
            lastModificationIteration = modifyBackup;
        }

        // PrimarySample Public Data
        int64_t lastModificationIteration = 0;
        Float valueBackup = 0;
        int64_t modifyBackup = 0;
    };

    struct PathSample {
        // PathSample Public Methods
        void Swap(PathSample &ps) {
        	int r = ps.t;
        	ps.t = t; t = r;
        	r = ps.s;
        	ps.s = s; s = r;
        	cv.swap(ps.cv);
        	lv.swap(ps.lv);
        }

        // PathSample Public Data
        int t;
        int s;
        std::vector<Vertex> cv;
        std::vector<Vertex> lv;
    };

    // GMLTSampler Private Methods
    void EnsureReady(int streamIndex, int index);
    std::vector<PrimarySample> &GetX(int stream){
    	if(stream == 0){
    		return camX;
    	}
    	else if(stream == 1){
    		return lightX;
    	}
    	else if(stream == 2){
    		return conX;
    	}
    	else{
    		Error("Non-existing stream specified.");
    		return camX;
    	}
    }

    // GMLTSampler Private Data
    RNG rng;
    const Float sigma, largeStepProbability;
    const int streamCount;
    std::vector<PrimarySample> camX;
    std::vector<PrimarySample> lightX;
    std::vector<PrimarySample> conX;
    PathSample current;
    PathSample proposed;
    int64_t currentIteration = 0;
    bool largeStep = true;
    bool crossover = false;
    int64_t lastLargeStepIteration = 0;
    int streamIndex, sampleIndex;
};

// Crossover Declarations
class Crossover{
public:
    Crossover() = default;
    virtual ~Crossover() = default;
    virtual bool Use(Float u, GMLTSampler &s1, GMLTSampler &s2, Float &probFactor) = 0;

};

class OnePointPSSCrossover : public Crossover{
	bool Use(Float u, GMLTSampler &s1, GMLTSampler &s2, Float &probFactor);
};

class OnePointPathSpaceCrossover : public Crossover{
public:
	bool Use(Float u, GMLTSampler &sam1, GMLTSampler &sam2, Float &probFactor);
};

class CopyCrossover : public Crossover{
public:
	bool Use(Float u, GMLTSampler &sam1, GMLTSampler &sam2, Float &probFactor);
};

class ArithmeticCrossover : public Crossover{
public:
	bool Use(Float u, GMLTSampler &sam1, GMLTSampler &sam2, Float &probFactor);
};

class BlendCrossover : public Crossover{
public:
	bool Use(Float u, GMLTSampler &sam1, GMLTSampler &sam2, Float &probFactor);
};

//CreateCrossover
std::shared_ptr<Crossover> CreateCrossover(std::string crossover);


// GMLTCrossoverSampler Declarations
class GMLTCrossoverSampler{
  public:
    // GMLTCrossoverSampler Public Methods
    GMLTCrossoverSampler(int mutationsPerPixel, std::vector<int> rngSequenceIndex, int seed, Float sigma,
               Float largeStepProbability, std::shared_ptr<Crossover> cross, Float crossoverProbability, int streamCount):
          rng(rngSequenceIndex[0]*seed),
		  crossover(std::move(cross)),
		  crossoverProbability(crossoverProbability)
  	{
    	int n = rngSequenceIndex.size();
    	samplers.reserve(n);
    	for (int i = 0; i < n; ++i) {
    	    samplers.emplace_back(mutationsPerPixel, rngSequenceIndex.at(i)*seed, sigma, largeStepProbability, streamCount);
    	}
    }
    GMLTCrossoverSampler(const GMLTCrossoverSampler &) = default;
    GMLTCrossoverSampler(GMLTCrossoverSampler &&) = default;
    void StartIteration();
    void Accept(int index);
    void Reject(int index);

    GMLTSampler &GetSampler(int index) { return samplers.at(index); }
    bool ExecCrossover(int* i, Float &probFactor);

  protected:
    // GMLTCrossoverSampler Private Methods

    // GMLTCrossoverSampler Private Data
    RNG rng;
    std::vector<GMLTSampler> samplers;
    const Float crossoverProbability;
    bool doCrossover = false;
    std::shared_ptr<Crossover> crossover;
};

// GMLT Declarations
class GMLTIntegrator : public Integrator {
  public:
    // GMLTIntegrator Public Methods
    GMLTIntegrator(std::shared_ptr<const Camera> camera, bool random, int maxDepth,
                  int nBootstrap, int nChains, int nChainsPerThread, int mutationsPerPixel,
                  Float sigma, Float largeStepProbability, std::string crossoverString, Float crossoverProbability)
        : camera(camera),
		  random(random),
          maxDepth(maxDepth),
          nBootstrap(nBootstrap),
          nChains(nChains),
		  nChainsPerThread(nChainsPerThread),
          mutationsPerPixel(mutationsPerPixel),
          sigma(sigma),
          largeStepProbability(largeStepProbability/(1-crossoverProbability)),
		  crossoverProbability( ((nChainsPerThread/2.0) * crossoverProbability) / (1 + (nChainsPerThread/2.0 - 1) * crossoverProbability) ),
		  actualLargeStepProbability(largeStepProbability),
		  actualCrossoverProbability(crossoverProbability),
  	  	  crossoverString(crossoverString)
		  {
			crossover = CreateCrossover(crossoverString);
		  }
    void Render(const Scene &scene);
    Spectrum L(const Scene &scene, MemoryArena &arena,
                   const std::unique_ptr<Distribution1D> &lightDistr,
                   const std::unordered_map<const Light *, size_t> &lightToIndex,
                   GMLTSampler &sampler, int k, Point2f *pRaster);

  private:
    // GMLTIntegrator Private Methods
    void RegisterStatistics(int nTotalMutations, int nThreads);

    // GMLTIntegrator Private Data
    std::shared_ptr<const Camera> camera;
    const bool random;
    const int maxDepth;
    const int nBootstrap;
    const int nChains;
    const int nChainsPerThread;
    const int mutationsPerPixel;
    const Float sigma, largeStepProbability, crossoverProbability, actualLargeStepProbability, actualCrossoverProbability;
    std::shared_ptr<Crossover> crossover;
    std::string crossoverString;
};

GMLTIntegrator *CreateGMLTIntegrator(const ParamSet &params,
                                   std::shared_ptr<const Camera> camera);
}  // namespace pbrt

#endif  // PBRT_INTEGRATORS_GMLT_H
