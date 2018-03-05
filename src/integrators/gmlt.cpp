
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


// integrators/gmlt.cpp*
#include "integrators/gmlt.h"
#include "integrators/bdpt.h"
#include "integrators/reversemapping.h"
#include "scene.h"
#include "film.h"
#include "sampler.h"
#include "integrator.h"
#include "camera.h"
#include "stats.h"
#include "filters/box.h"
#include "paramset.h"
#include "sampling.h"
#include "progressreporter.h"

namespace pbrt {

STAT_CROSSOVER("General/Crossover", cross);
STAT_COUNTER("General/Max Depth", maxdepth);
STAT_DOUBLE("General/Crossover Probability", crossProb);
STAT_DOUBLE("General/Large Step Probability", largeStepProb);
STAT_COUNTER("General/Bootstrap Samples", bootstrapSamples);
STAT_COUNTER("General/Mutations per pixel", mutperpix);
STAT_COUNTER("General/Total Mutations", totalMut);
STAT_COUNTER("General/Threads", threads);
STAT_COUNTER("General/Chains", chains);
STAT_COUNTER("General/Chains Per Thread", chainsPerThread);
STAT_PERCENT("Integrator/Acceptance rate", acceptedMutations, totalMutations);
STAT_PERCENT("Integrator/Crossover rate", crossoverMutations, totalMutations2);
STAT_PERCENT("Integrator/Crossover Acceptance rate", acceptedCrossovers, totalCrossovers);
STAT_PERCENT("Integrator/Zero-radiance crossover", zeroCrossovers, totalCrossovers2);

// GMLTSampler Constants
static const int cameraStreamIndex = 0;
static const int lightStreamIndex = 1;
static const int connectionStreamIndex = 2;
static const int nSampleStreams = 3;

// GMLTSampler Method Definitions
Float GMLTSampler::Get1D() {
    ProfilePhase _(Prof::GetSample);
    int index = GetNextIndex();
    EnsureReady(streamIndex,index);
    return GetXi(streamIndex, index);
}

Point2f GMLTSampler::Get2D() { return {Get1D(), Get1D()}; }

std::unique_ptr<Sampler> GMLTSampler::Clone(int seed) {
    LOG(FATAL) << "GMLTSampler::Clone() is not implemented";
    return nullptr;
}

void GMLTSampler::StartIteration() {
    currentIteration++;
    largeStep = rng.UniformFloat() < largeStepProbability;
    crossover = false;
}

void GMLTSampler::EnsureReady(int stream, int index) {
    // Enlarge _GMLTSampler::X_ if necessary and get current $\VEC{X}_i$
    if (index >= GetX(stream).size()) GetX(stream).resize(index + 1);
    PrimarySample &Xi = GetX(stream)[index];

    // Reset $\VEC{X}_i$ if a large step took place in the meantime
    if (Xi.lastModificationIteration < lastLargeStepIteration) {
        Xi.value = rng.UniformFloat();
        Xi.lastModificationIteration = lastLargeStepIteration;
    }

    // Apply remaining sequence of mutations to _sample_
    if(!crossover){
     	Xi.Backup();
        if (largeStep) {
        	Xi.value = rng.UniformFloat();
        } else {
        	int64_t nSmall = currentIteration - Xi.lastModificationIteration;
        	// Apply _nSmall_ small step mutations

        	// Sample the standard normal distribution $N(0, 1)$
        	Float normalSample = Sqrt2 * ErfInv(2 * rng.UniformFloat() - 1);

        	// Compute the effective standard deviation and apply perturbation to
        	// $\VEC{X}_i$
        	Float effSigma = sigma * std::sqrt((Float)nSmall);
        	Xi.value += normalSample * effSigma;
        	Xi.value -= std::floor(Xi.value);
    	}
    }
    Xi.lastModificationIteration = currentIteration;
}

void GMLTSampler::Accept() {
    if (largeStep) lastLargeStepIteration = currentIteration;
    current.Swap(proposed);
}

void GMLTSampler::Reject() {
    for (auto &Xi : camX)
        if (Xi.lastModificationIteration == currentIteration) Xi.Restore();
    for (auto &Xi : lightX)
        if (Xi.lastModificationIteration == currentIteration) Xi.Restore();
    for (auto &Xi : conX)
        if (Xi.lastModificationIteration == currentIteration) Xi.Restore();
    --currentIteration;
}

void GMLTSampler::StartStream(int index) {
    CHECK_LT(index, streamCount);
    streamIndex = index;
    sampleIndex = 0;
}

int GMLTSampler::GetStreamForIndex(int index, int &newIndex){
	int camS = camX.size();
	int clS = lightX.size() + camS;
	if(index >= camS){
		if(index >= clS){
			newIndex = index - clS;
			return 2;
		}
		else{
			newIndex = index - camS;
			return 1;
		}
	}
	else{
		newIndex = index;
		return 0;
	}
}

Float GMLTSampler::GetXi(int stream, int index){
	return GetX(stream)[index].value;
}

void GMLTSampler::SetXi(int stream, int index, Float value) {
	GetX(stream)[index].Backup();
	GetX(stream)[index].value = value;
}

// GMLTCrossoverSampler Method Definitions
void GMLTCrossoverSampler::StartIteration() {
	int n = samplers.size();
	for (int k = 0; k < n; ++k) {
	    samplers.at(k).StartIteration();
	}
    if(n > 1) doCrossover = rng.UniformFloat() < crossoverProbability;
}

void GMLTCrossoverSampler::Accept(int index) {
	samplers.at(index).Accept();
}

void GMLTCrossoverSampler::Reject(int index) {
	samplers.at(index).Reject();
}

bool GMLTCrossoverSampler::ExecCrossover(int* i, Float &probFactor) {
    int n = samplers.size();
    if( n == 1 ) return false;

    i[0] = n*rng.UniformFloat();
    i[1] = (n-1)*rng.UniformFloat();
    if(i[1]>=i[0]) i[1]++;

    GMLTSampler &s1 = samplers.at(i[0]);
    GMLTSampler &s2 = samplers.at(i[1]);
    Float u = rng.UniformFloat();

    bool didCrossover = false;
    if(doCrossover && s1.IsNoneZeroIteration() && s2.IsNoneZeroIteration()){
    	didCrossover = crossover.get()->Use(u, s1, s2, probFactor);
    }
    s1.SetCrossover(didCrossover);
    s2.SetCrossover(didCrossover);
    return didCrossover;
}

// GMLT Method Definitions
Spectrum GMLTIntegrator::L(const Scene &scene, MemoryArena &arena,
                          const std::unique_ptr<Distribution1D> &lightDistr,
                          const std::unordered_map<const Light *, size_t> &lightToIndex,
                          GMLTSampler &sampler, int depth, Point2f *pRaster) {
    sampler.StartStream(cameraStreamIndex);
    // Determine the number of available strategies and pick a specific one
    int t, s, nStrategies;
    if (depth == 0) {
        nStrategies = 1;
        s = 0;
        t = 2;
    } else {
        nStrategies = depth + 2;
        s = std::min((int)(sampler.Get1D() * nStrategies), nStrategies - 1);
        t = nStrategies - s;
    }
    sampler.SetProposedT(t);
    sampler.SetProposedS(s);

    // Generate a camera subpath with exactly _t_ vertices
    Vertex * cv = sampler.GetProposedCV(t);
    Bounds2f sampleBounds = (Bounds2f)camera->film->GetSampleBounds();
    *pRaster = sampleBounds.Lerp(sampler.Get2D());
    if (GenerateCameraSubpath(scene, sampler, arena, t, *camera, *pRaster, cv) != t)
        return Spectrum(0.f);

    // Generate a light subpath with exactly _s_ vertices
    sampler.StartStream(lightStreamIndex);
    Vertex * lv = sampler.GetProposedLV(s);
    if (GenerateLightSubpath(scene, sampler, arena, s, cv[0].time(),
                             *lightDistr, lightToIndex, lv) != s)
        return Spectrum(0.f);

    // Execute connection strategy and return the radiance estimate
    sampler.StartStream(connectionStreamIndex);
    return ConnectBDPT(scene, lv, cv, s, t, *lightDistr,
                       lightToIndex, *camera, sampler, pRaster) *
           nStrategies;
}

void GMLTIntegrator::Render(const Scene &scene) {
    /*Create local path distribution class*/
    class PathDistribution {
    	public:
        // PathDistribution Public Methods
    	PathDistribution(const Float *f, int maxDepth, int n) : distr(f,n) {
    		int amtPerDepth = n/(maxDepth+1);
    		std::vector< std::vector< Float > > fs(maxDepth+1, std::vector<Float>(amtPerDepth));

    		for (int i = 0; i < n; ++i) {
    			int depth = i % (maxDepth + 1);
    			int j = i/(maxDepth + 1);
    			fs[depth][j] = f[i];
    		}

    		depthDistr.reserve(maxDepth+1);
    		for (int i = 0; i <= maxDepth; ++i) {
    			depthDistr.emplace_back(fs[i].data(), amtPerDepth);
    		}
        }

        int SampleDiscrete(Float u) const {
        	return distr.SampleDiscrete(u);
        }

        int SampleDiscrete(Float u, int depth) const {
        	int s = depthDistr[depth].SampleDiscrete(u);
        	return s * (depthDistr.size()) + depth;
        }

        // PathDistribution Private Data
    	//private:
        std::vector<Distribution1D> depthDistr;
        Distribution1D distr;
    };

    /****************
     * Start Method
     ****************/

    std::unique_ptr<Distribution1D> lightDistr =
        ComputeLightPowerDistribution(scene);

    int seed = 1;
    if(random){
    	srand( time(NULL) );
    	seed = rand();
    }

    // Compute a reverse mapping from light pointers to offsets into the
    // scene lights vector (and, equivalently, offsets into
    // lightDistr). Added after book text was finalized; this is critical
    // to reasonable performance with 100s+ of light sources.
    std::unordered_map<const Light *, size_t> lightToIndex;
    for (size_t i = 0; i < scene.lights.size(); ++i)
        lightToIndex[scene.lights[i].get()] = i;

    // Generate bootstrap samples and compute normalization constant $b$
    int nBootstrapSamples = nBootstrap * (maxDepth + 1);
    std::vector<Float> bootstrapWeights(nBootstrapSamples, 0);
    if (scene.lights.size() > 0) {
        ProgressReporter progress(nBootstrap / 256,
                                  "Generating bootstrap paths");
        std::vector<MemoryArena> bootstrapThreadArenas(MaxThreadIndex());
        int chunkSize = Clamp(nBootstrap / 128, 1, 8192);
        ParallelFor([&](int i) {
            // Generate _i_th bootstrap sample
            MemoryArena &arena = bootstrapThreadArenas[ThreadIndex];
            for (int depth = 0; depth <= maxDepth; ++depth) {
            	int index = i * (maxDepth + 1) + depth;
            	int rngIndex = index*seed;
                GMLTSampler sampler(mutationsPerPixel, rngIndex, sigma, largeStepProbability, nSampleStreams);
                Point2f pRaster;
                bootstrapWeights[index] =
                    L(scene, arena, lightDistr, lightToIndex, sampler, depth, &pRaster).y();
                arena.Reset();
            }
            if ((i + 1) % 256 == 0) progress.Update();
        }, nBootstrap, chunkSize);
        progress.Done();
    }
    PathDistribution bootstrap(&bootstrapWeights[0], maxDepth, nBootstrapSamples);
    Float b = bootstrap.distr.funcInt * (maxDepth + 1);

    // Run _nChains_ Markov chains in parallel
    Film &film = *camera->film;
    int64_t nTotalMutations =
        (int64_t)mutationsPerPixel * (int64_t)film.GetSampleBounds().Area();
    if (scene.lights.size() > 0) {
        const int progressFrequency = 32768;
        ProgressReporter progress(nTotalMutations / progressFrequency,
                                  "Rendering");

        //round up
        int nThreads = (nChains+nChainsPerThread-1)/nChainsPerThread;

        //Create Statistics
        RegisterStatistics(nTotalMutations, nThreads);

        ParallelFor([&](int i) {
        	int nThreadChains = std::min((i+1) * nChainsPerThread, nChains) - i * nChainsPerThread;

            /*int64_t nThreadMutations =
            	std::min((i + 1) * nTotalMutations / nThreads, nTotalMutations) - i * nTotalMutations / nThreads;
            int64_t nChainMutations = nThreadMutations/nThreadChains;*/

        	int64_t nThreadMutations = std::min((i + 1) * nTotalMutations / nThreads, nTotalMutations) - i * nTotalMutations / nThreads;
        	int64_t todoMutations = nThreadMutations;
        	bool prevprogresscheck = false;

            // Follow {i}th Markov chain for _nChainMutations_
            MemoryArena arena;

            // Select initial state from the set of bootstrap samples
            std::vector<int> bootstrapIndex(nThreadChains);
            RNG rng(i*seed);
            bootstrapIndex[0] = bootstrap.SampleDiscrete(rng.UniformFloat());
            int depth = bootstrapIndex[0] % (maxDepth + 1);
            for (int k = 1; k < nThreadChains; ++k) {
            	bootstrapIndex[k] = bootstrap.SampleDiscrete(rng.UniformFloat(),depth);
			}

            // Initialize local variables for selected state
            GMLTCrossoverSampler sampler(mutationsPerPixel, bootstrapIndex, seed, sigma,
                    largeStepProbability, crossover, crossoverProbability, nSampleStreams);

            std::vector<Point2f> pCurrent(nThreadChains);
            std::vector<Spectrum> LCurrent(nThreadChains);
            for (int k = 0; k < nThreadChains; ++k) {
                LCurrent[k] = L(scene, arena, lightDistr, lightToIndex, sampler.GetSampler(k), depth, &pCurrent[k]);
                sampler.Accept(k);//Swaps path from proposed to current
            }

            // Run the Markov chain for _nChainMutations_ steps
            while (todoMutations > nThreadChains) {
            	sampler.StartIteration();
            	int crossoverIndices[2];
            	Float probFactor;
            	bool didCrossover = sampler.ExecCrossover(crossoverIndices, probFactor);

            	//Iterate over crossover chains
            	int mutationsDone = 0;
            	if(didCrossover){
            		Point2f pProposed[2];
            	    Spectrum LProposed[2];
            	    Float acceptCross[2];
            	    for (int k = 0; k < 2; ++k) {
            	    	int index = crossoverIndices[k];
            	        LProposed[k] = L(scene, arena, lightDistr, lightToIndex, sampler.GetSampler(index), depth, &pProposed[k]);

            	        // Compute acceptance probability for proposed sample
            	        acceptCross[k] = std::min((Float)1, (LProposed[k].y() / LCurrent[index].y()) * probFactor);

            	        // Splat both current and proposed samples to _film_
            	        if (acceptCross[k] > 0)
            	        	film.AddSplat(pProposed[k], LProposed[k] * acceptCross[k] / LProposed[k].y());
            	        else ++zeroCrossovers;
            	       	film.AddSplat(pCurrent[index], LCurrent[index] * (1 - acceptCross[k]) / LCurrent[index].y());
            	       	++totalCrossovers2;
            	    }

            	    // Accept or reject the proposal
            	    if (rng.UniformFloat() < acceptCross[0] && rng.UniformFloat() < acceptCross[1]) {
            	    	pCurrent[crossoverIndices[0]] = pProposed[0];
            	        LCurrent[crossoverIndices[1]] = LProposed[1];
            	        sampler.Accept(crossoverIndices[0]);
            	        sampler.Accept(crossoverIndices[1]);
            	        acceptedMutations += 2;
            	        acceptedCrossovers += 2;
            	    } else{
            	      	sampler.Reject(crossoverIndices[0]);
            	        sampler.Reject(crossoverIndices[1]);
            	    }
            	    crossoverMutations += 2;
            	    totalCrossovers += 2;
            	    mutationsDone = 2;
            	}
            	else{
					//Iterate over all chains if no crossover
					for (int k = 0; k < nThreadChains; ++k) {
						Point2f pProposed;
						Spectrum LProposed = L(scene, arena, lightDistr, lightToIndex, sampler.GetSampler(k), depth, &pProposed);

						// Compute acceptance probability for proposed sample
						Float accept = std::min((Float)1, LProposed.y() / LCurrent[k].y());

						// Splat both current and proposed samples to _film_
						if (accept > 0)
							film.AddSplat(pProposed, LProposed * accept / LProposed.y());
						film.AddSplat(pCurrent[k], LCurrent[k] * (1 - accept) / LCurrent[k].y());

						// Accept or reject the proposal
						if (rng.UniformFloat() < accept) {
							pCurrent[k] = pProposed;
							LCurrent[k] = LProposed;
							sampler.Accept(k);
							++acceptedMutations;
						} else{
							sampler.Reject(k);
						}
					}
					mutationsDone = nThreadChains;
            	}
            	totalMutations += mutationsDone;
            	totalMutations2 += mutationsDone;
            	todoMutations -= mutationsDone;
            	int mod = (i*nTotalMutations/nChains + (nThreadMutations - todoMutations)) % progressFrequency;
            	if ( !prevprogresscheck && mod < nThreadChains ){
            	    progress.Update();
            	    prevprogresscheck = true;
            	}
            	if ( mod > nThreadChains ) prevprogresscheck = false;
            	arena.Reset();
            }
            for (int k = 0; k < todoMutations; ++k) {
				Point2f pProposed;
				Spectrum LProposed = L(scene, arena, lightDistr, lightToIndex, sampler.GetSampler(k), depth, &pProposed);

				// Compute acceptance probability for proposed sample
				Float accept = std::min((Float)1, LProposed.y() / LCurrent[k].y());

				// Splat both current and proposed samples to _film_
				if (accept > 0)
					film.AddSplat(pProposed, LProposed * accept / LProposed.y());
				film.AddSplat(pCurrent[k], LCurrent[k] * (1 - accept) / LCurrent[k].y());

				// Accept or reject the proposal
				if (rng.UniformFloat() < accept) {
					pCurrent[k] = pProposed;
					LCurrent[k] = LProposed;
					sampler.Accept(k);
					++acceptedMutations;
				} else{
					sampler.Reject(k);
				}
            }
            totalMutations += todoMutations;
            totalMutations2 += todoMutations;
        }, nThreads);
        progress.Done();
    }

    // Store final image computed with GMLT
    camera->film->WriteImage(b / mutationsPerPixel);
}

std::string to_string(double x){
  std::ostringstream ss;
  ss << x;
  return ss.str();
}

void GMLTIntegrator::RegisterStatistics(int nTotalMutations, int nThreads){
	if(crossoverString == "copy") cross = 3;
	else if(crossoverString == "onepointpss") cross = 1;
	else if(crossoverString == "onepointpath") cross = 2;
	else if(crossoverString == "arithmetic") cross = 4;
	else if(crossoverString == "blend") cross = 5;
	maxdepth = maxDepth;
	crossProb = actualCrossoverProbability;
    largeStepProb = actualLargeStepProbability;
    bootstrapSamples = nBootstrap;
    mutperpix = mutationsPerPixel;
	totalMut = nTotalMutations;
	threads = nThreads;
	chains = nChains;
	chainsPerThread = nChainsPerThread;
}

GMLTIntegrator *CreateGMLTIntegrator(const ParamSet &params,
                                   std::shared_ptr<const Camera> camera) {
	bool random = params.FindOneBool("random", false);
    int maxDepth = params.FindOneInt("maxdepth", 5);
    int nBootstrap = params.FindOneInt("bootstrapsamples", 100000);
    int64_t nChains = params.FindOneInt("chains", 1000);
    int nChainsPerThread = params.FindOneInt("chainsPerThread", 8);
    int mutationsPerPixel = params.FindOneInt("mutationsperpixel", 100);
    Float largeStepProbability = params.FindOneFloat("largestepprobability", 0.3f);
    Float crossoverProbability = params.FindOneFloat("crossoverprobability", 0.3f);
    Float sigma = params.FindOneFloat("sigma", .01f);
    if (PbrtOptions.quickRender) {
        mutationsPerPixel = std::max(1, mutationsPerPixel / 16);
        nBootstrap = std::max(1, nBootstrap / 16);
    }
    std::string crossover = params.FindOneString("crossover","onepointpss");

    return new GMLTIntegrator(camera, random, maxDepth, nBootstrap, nChains, nChainsPerThread,
                             mutationsPerPixel, sigma, largeStepProbability, crossover, crossoverProbability);
}

std::shared_ptr<Crossover> CreateCrossover(std::string crossover){
    std::shared_ptr<Crossover> c;

    if (crossover == "onepointpss") {
        c.reset(new OnePointPSSCrossover());
    } else if (crossover == "onepointpath") {
        c.reset(new OnePointPathSpaceCrossover());
    } else if (crossover == "copy") {
        c.reset(new CopyCrossover());
    } else if (crossover == "arithmetic") {
        c.reset(new ArithmeticCrossover());
    } else if (crossover == "blend") {
        c.reset(new BlendCrossover());
    }else {
        Error("Crossover \"%s\" unknown.", crossover.c_str());
    }
    return c;
}

bool OnePointPSSCrossover::Use(Float u, GMLTSampler &s1, GMLTSampler &s2, Float &probFactor){
        if(s1.GetLength() != s2.GetLength()) return false;

        int crossoverPoint = u * s1.GetLength();
        for (int i = 0; i < s1.GetLength(); ++i) {
        	int i1, i2;
        	int stream1 = s1.GetStreamForIndex(i, i1);
        	int stream2 = s2.GetStreamForIndex(i, i2);
        	if(i >= crossoverPoint){
            	//Swap
                Float Xi = s1.GetXi(stream1,i1);
                s1.SetXi(stream1, i1, s2.GetXi(stream2, i2));
                s2.SetXi(stream2, i2, Xi);
            }
            else{
            	//Copy
                s1.SetXi(stream1, i1, s1.GetXi(stream1, i1));
                s2.SetXi(stream2, i2, s2.GetXi(stream2, i2));
            }
        }

        probFactor = 1;
        return true;
}

bool OnePointPathSpaceCrossover::Use(Float u, GMLTSampler &sam1, GMLTSampler &sam2, Float &probFactor){
		if(sam1.GetDepth() != sam2.GetDepth()) return false;
		int depth = sam1.GetDepth();
		if(depth > 1){
			//crossover pathsamples
			int crossoverPoint = u * (depth- 1);
			int t1 = sam1.GetCurrentT();
			int s1 = sam1.GetCurrentS();
			int t2 = sam2.GetCurrentT();
			int s2 = sam2.GetCurrentS();
			std::vector<Vertex> &cv1 = sam1.GetCurrentCV();
			std::vector<Vertex> &lv1 = sam1.GetCurrentLV();
			std::vector<Vertex> &cv2 = sam2.GetCurrentCV();
			std::vector<Vertex> &lv2 = sam2.GetCurrentLV();
			for (int i = 2; i < depth + 2; ++i) {
				if(i >= crossoverPoint){
					//Swap
					Vertex &v = i < t1 ? cv1[i] : lv1[(s1-1) - (i-t1)];
					Vertex vi = v;

					if(i < t2){
						v = cv2[i];
						cv2[i] = vi;
					}
					else{
						v = lv2[(s2-1) - (i-t2)];
						lv2[(s2-1) - (i-t2)] = vi;
					}
				 }
			}
		}

		//Reverse map path samples to primary samples
		ReverseMapping rm;
		rm.ReverseMap(sam1);
		rm.ReverseMap(sam2);

		probFactor = 1;
        return true;
}

bool CopyCrossover::Use(Float u, GMLTSampler &sam1, GMLTSampler &sam2, Float &probFactor){
		//Reverse map path samples to primary samples
		ReverseMapping rm;
		rm.ReverseMap(sam1);
		rm.ReverseMap(sam2);

		probFactor = 1;
        return true;
}

bool ArithmeticCrossover::Use(Float u, GMLTSampler &s1, GMLTSampler &s2, Float &probFactor){
        if(s1.GetLength() != s2.GetLength()) return false;

        int v = 1-u;
        for (int i = 0; i < s1.GetLength(); ++i) {
        	int i1, i2;
        	int stream1 = s1.GetStreamForIndex(i, i1);
        	int stream2 = s2.GetStreamForIndex(i, i2);

        	Float xi = s1.GetXi(stream1, i1);
        	Float yi = s2.GetXi(stream2, i2);

        	s1.SetXi(stream1, i1, xi*u + yi*v);
        	s2.SetXi(stream2, i2, xi*v + yi*u);
        }

        probFactor = 1;
        return true;
}

bool BlendCrossover::Use(Float u, GMLTSampler &s1, GMLTSampler &s2, Float &probFactor){
        if(s1.GetLength() != s2.GetLength()) return false;

        RNG rng(u);
        probFactor = 1;
        Float sum = 0;
        for (int i = 0; i < s1.GetLength(); ++i) {
        	int i1, i2;
        	int stream1 = s1.GetStreamForIndex(i, i1);
        	int stream2 = s2.GetStreamForIndex(i, i2);
        	Float g1 = rng.UniformFloat() - 0.5;
        	Float g2 = rng.UniformFloat() - 0.5;

        	Float xi = s1.GetXi(stream1, i1);
        	Float xj = s2.GetXi(stream2, i2);

        	Float yi = xi*(1-g1) + xj*g1;
        	Float yj = xi*g2 + xj*(1-g2);

        	yi = std::min( (Float) 1 , std::max( (Float) 0, yi) );
        	yj = std::min( (Float) 1 , std::max( (Float) 0, yj) );

        	if(yi > 1 || yi < 0 || yj > 1 || yj < 0 ){
        		printf("yi: %f, yj: %f", yi, yj);
        	}

        	s1.SetXi(stream1, i1, yi);
        	s2.SetXi(stream2, i2, yj);
        	Float dx = std::abs(xi-xj);
        	Float dy = std::abs(yi-yj);
        	if(dx == 0 || dy == 0) probFactor *= 0;
        	else {
        		sum += g1 + g2;
        		probFactor *= dy/dx;
        	}
        }

        return true;
}

}  // namespace pbrt
