// FGrnLib.h

#if !defined FGRNLIB_H_
#define	FGRNLIB_H_

#include <algorithm>
#include <vector>
#include <cmath>
#include <random>
#include <string>
#include <fstream>
#include <sstream>

#include <iomanip>  // for print double precision

// Visual Studio Warning
//#pragma warning( disable : 4290 )	// exception warning in function:  void f throw(int) {...}

namespace FGrn {

#define PRECISION_FACTOR	10000.0	
#define MAX_ALGORITHM_LOOP	5

	enum {
		ERROR_IN_PARAMETER_EXCEPTION = 1,
		ERROR_IN_FILE_OPEN,
		ERROR_IN_DIVISION_BY_ZERO
	};

	enum LearningState {
		NOT_LEARNING = 0,
		LEARNING,
		BATCH_LEARNING
	};

	double approximate(double d, double pf = PRECISION_FACTOR);

	class lMatrix;
	class sMatrix;

	class lVector {
		size_t size = 0, deltaIndex = 0;
		bool normalized = true, uniform = false;
		std::vector<double> vector;
		void check(bool norm);
		lVector operator*(double d) const;
	public:
		lVector() {}
		lVector(size_t dim, size_t index = 0);
		lVector(const std::vector<double> &v, bool norm = false);
		lVector(const std::initializer_list<double> l);

		bool isNormal() const { return normalized; }
		bool isUniform() const { return uniform; }
		bool isSimilar(const lVector &v) const; // Solo per vettori normalizzati (per il momento)

		size_t getDelta() const { return deltaIndex; }
		size_t getSize() const { return size; }
		double getSum() const;
		std::vector<double> getVector() const;
		void setVector(const std::vector<double> &v);
		
		void normalize();
		lMatrix rowColProd(const lVector &v) const;

		double operator[](int i) const { return exp2(vector[i]); }
		lVector& operator/=(double v);
		lVector operator*(const lVector &v) const;
		lVector operator*(const lMatrix &m) const;
		
		friend std::ostream& operator<<(std::ostream &stream, const lVector &ob);
		friend std::istream& operator>>(std::istream &stream, lVector &ob);
		
	};

	template<class T> class  Matrix {
	protected:
		size_t rows = 0, cols = 0, colIndex = 0;
		std::vector<T> matrix;
	public:
		size_t getNRows() const { return rows; }
		size_t getNColumns() const { return cols; }
		size_t isColumn() const { return colIndex; }
		T operator[](int i) const { return matrix[i]; }
		friend std::ostream& operator<<(std::ostream &stream, const Matrix<T> &ob);
		friend std::istream& operator>>(std::istream &stream, Matrix<T> &ob);
	};

	class lMatrix : public Matrix<lVector> {
	public:
		lMatrix() {}
		lMatrix(size_t r, size_t c, bool casual = false);
		lMatrix(const std::vector<lVector> &m, size_t index = 0);
		lMatrix(const sMatrix &m);
		
		bool isSimilar(const lMatrix &m) const;
		//void setMatrix(const std::vector<lVector> &m);
		sMatrix normalize();
		
		//lVector operator[](int i) const { return matrix[i]; }
		//lMatrix& operator/=(double d);
		lVector operator*(const lVector &v) const;
		lMatrix operator*(const lMatrix &m) const;
	};

	class sMatrix : public Matrix<std::vector<double>> {
	public:
		sMatrix() {}
		sMatrix(size_t r, size_t c); // tutto a zero;
		sMatrix(const lMatrix &m);
		lVector getRow(size_t index) const;
		// std::vector<double> operator[](int i) const { return matrix[i]; }
		sMatrix& operator+=(const sMatrix &m);
	};

	///Abstract class
	class Element {
	public:
		virtual void reset() = 0;
		virtual void update(bool fromDown) = 0;
	};

	/*
	Class: Link
	Purpose: Define a variable in a DICA scheme.
	*/
	class Link {
		size_t cardinality;
		Element *parent = nullptr, *child = nullptr;
		lVector forwardMessage;
		lVector backwardMessage;
	public:
		Link(size_t d = 2);
		void setElement(Element &e, bool dad) { (dad) ? parent = &e : child = &e; }
		void setForwardMessage(const lVector &f);
		void setBackwardMessage(const lVector &b);
		Element* getParent() const { return parent; }
		Element* getChild() const { return child; }
		size_t getCardinality() const { return cardinality; }
		const lVector& getForwardMessage() const { return forwardMessage; }
		const lVector& getBackwardMessage() const { return backwardMessage; }
		void reset();
	};

	/*
	Class: SISOBlock
	Purpose: Define a block SISO in a DICA scheme.
	*/
	class SISOBlock : public Element {
		lMatrix matrix;
		bool casualityMatrix;
		sMatrix tempLearningMatrix;
		Link *inputLink, *outputLink;
		LearningState actualState = NOT_LEARNING;
		std::vector<lVector> listInput, listOutput;
		void initialize();
	public:
		SISOBlock() {}
		SISOBlock(Link &in, Link &out, bool casual = false);
		void setElement(Link &in, Link &out, bool casual = false);
		void setMatrix(const lMatrix &m);
		void setState(LearningState state);
		lMatrix getMatrix() const { return matrix; }
		void putsOut(size_t index);
		void update(bool fromDown);
		bool learn(size_t loops = MAX_ALGORITHM_LOOP);
		void reset();
	};

	/*
	Class: Diverter
	Purpose: Define a diverter in a DICA scheme.
	*/
	class Diverter : public Element {
		std::vector<Link*> inputs;
		std::vector<Link*> outputs;
		std::vector<std::vector<size_t>> inMatrices;
		lVector inMoltiplicator(size_t index);
		void outMoltiplicator(size_t index, const lVector &out);
		void initialize();
	public:
		Diverter(std::vector<Link> &in, std::vector<Link> &out);
		void update(bool fromDown) {}
		void execute();
		void reset() {}
	};

	/*
	Class: Source
	Purpose: Define a source in a DICA scheme.
	*/
	class Source : public Element {
		Link *outputLink;
		lVector prior;
		LearningState actualState = NOT_LEARNING;
		std::vector<double> tempLearningPrior;
	public:
		Source(Link &out, const lVector *p = nullptr);
		void setOut(lVector v) { outputLink->setForwardMessage(v); }
		void setState(LearningState state);
		lVector getPrior() { return prior; }
		void update(bool fromDown);
		void learn();
		void reset();
	};

	/*
	Class: Supervisor
	Purpose: Define a supervisor in the normal factor graph.
	
	Da modificare opportunamente
	/
	class Supervisor : public Element {
		size_t cardinality = 0;
		std::vector<std::vector<Source>> sources;
		std::vector<Diverter> diverters;
		std::vector<std::vector<SISOBlock>> blocksLevels;
		std::vector<std::vector<Link>> lol;	// Links of levels (*2 becouse up and down link respect of sisoblock)
		void initialize();
	public:
		Supervisor(const std::vector<Link> &inputs);
		Supervisor(size_t nInput, size_t nLevel = 1, std::vector<size_t> *nSource = nullptr, std::vector<size_t> *cardOfInputs = nullptr, std::vector<std::vector<size_t>> *cardOfSources = nullptr, std::vector<std::vector<lVector>> *sourcesPrior = nullptr);


		Supervisor(std::vector<Source*> s, Diverter &d, std::vector<SISOBlock> &b, std::vector<Link> &out);		// one level
		Supervisor(std::vector<Source*> s, std::vector<Diverter*> d, std::vector<std::vector<SISOBlock*>> b, std::vector<Link*> out) : sources(s), diverters(d), blocksLevels(b), outputLinks(out) { initialize(); }
		void addOutputs(LearningState state, std::vector<probabilityVector> out);
		void update(bool fromDown) { }
		void learn(LearningState state, size_t epochs = MAX_ALGORITHM_LOOP, size_t loops = MAX_ALGORITHM_LOOP);
		void run();
		void reset() { cardinality = 0; }
		void save(const std::string &name);
	};
	//*/
	
}   // namespace FGrn

#endif	// FGRNLIB_H_
