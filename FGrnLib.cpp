// FGrnLib.cpp

#include "FGrnLib.h"

namespace FGrn {
	
	double approximate(double d, double pf) {
		return round(d * pf) / pf;
	}

	/* LVECTOR */

	void lVector::check(bool norm) {
		size_t i = 1;
		deltaIndex = 0;
		uniform = true;
		normalized = norm;
		double sum = 0.0;
		do {
			if (vector[i - 1] != vector[i]) {
				uniform = false;
				if (!vector[i - 1] || !vector[1]) {
					deltaIndex = (!vector[i - 1]) ? i + 1 : 1;
					for (++i; i < size; i++) {
						if (vector[i]) {
							sum = vector[deltaIndex - 1];
							deltaIndex = 0;
							break;
						}
					}
				}
				else sum = vector[i - 1] * i;
				break;
			}
			i++;
		} while (i < size);
		if (uniform) sum = vector[0] * size;
		else if (deltaIndex) sum = vector[deltaIndex - 1];
		else {
			#pragma omp parallel for reduction(+:sum)
			for (int j = i; j < size; j++) sum += vector[j];
		}
		#pragma omp parallel for
		for (int i = 0; i < size; i++) vector[i] = log2(vector[i]);
		if (sum == 1.0) normalized = true;
		else if (normalized) {
			sum = log2(sum);
			#pragma omp parallel for
			for (int i = 0; i < size; i++) vector[i] -= sum;
		}
	}

	lVector lVector::operator*(double d) const {
		if (!d) return *this;
		lVector v = *this;
		v.normalized = false;
		if (deltaIndex) v.vector[deltaIndex - 1] += d;
		else {
			#pragma omp parallel for
			for (int i = 0; i < size; i++) v.vector[i] += d;
		}
		return v;
	}

	lVector::lVector(size_t dim, size_t index) {
		if (dim < 2 || index > dim) throw ERROR_IN_PARAMETER_EXCEPTION;
		size = dim;
		if (index) {
			deltaIndex = index;
			vector.assign(dim, -INFINITY);
			vector[index - 1] = 0.0;
		}
		else {
			uniform = true;
			vector.assign(dim, log2((double)1 / dim));
		}
	}

	lVector::lVector(const std::vector<double> &v, bool norm) {
		if (v.size() < 2) throw ERROR_IN_PARAMETER_EXCEPTION;
		vector = v;
		size = vector.size();
		check(norm);
	}

	lVector::lVector(const std::initializer_list<double> l) {
		if (l.size() < 2) throw ERROR_IN_PARAMETER_EXCEPTION;
		vector = l;
		size = vector.size();
		check(false);
	}

	bool lVector::isSimilar(const lVector &v) const {
		if (size != v.size || deltaIndex != v.deltaIndex || uniform != v.uniform) return false;
		if (normalized && v.normalized && (deltaIndex || uniform)) return true;
		for (int i = 0; i < size; i++) {
			if (approximate(vector[i]) != approximate(v.vector[i])) return false;
		}
		return true;
	}

	double lVector::getSum() const {
		if (normalized) return 1.0;
		if (deltaIndex) return exp2(vector[deltaIndex - 1]);
		if (uniform) return exp2(vector[0]) * size;
		double sum = 0.0;
		#pragma omp parallel for reduction(+:sum)
		for (int i = 0; i < size; i++) sum += exp2(vector[i]);
		return sum;
	}

	std::vector<double> lVector::getVector() const {
		std::vector<double> v(size);
		#pragma omp parallel for
		for (int i = 0; i < size; i++) v[i] = exp2(vector[i]);
		return v;
	}

	void lVector::setVector(const std::vector<double> &v) {
		if (v.size() < 2) throw ERROR_IN_PARAMETER_EXCEPTION;
		vector = v;
		size = vector.size();
		check(false);
	}

	void lVector::normalize() {
		if (!normalized) {
			normalized = true;
			if (uniform) vector.assign(size, log2((double)1 / size));
			else if (deltaIndex) vector[deltaIndex - 1] = 0.0;
			else {
				double sum = 0.0;
				#pragma omp parallel for reduction(+:sum)
				for (int i = 0; i < size; i++) sum += exp2(vector[i]);
				sum = log2(sum);
				if (sum) {
					#pragma omp parallel for
					for (int j = 0; j < size; j++) vector[j] -= sum;
				}
			}
		}
	}

	lMatrix lVector::rowColProd(const lVector &v) const {
		std::vector<lVector> m(size);
		if (v.deltaIndex) {
			#pragma omp parallel for
			for (int i = 0; i < size; i++) {
				m[i] = lVector(v.size, v.deltaIndex);
				m[i].normalized = false;
				m[i].vector[v.deltaIndex - 1] = vector[i] + v.vector[v.deltaIndex - 1];
			}
			return lMatrix(m, v.deltaIndex);
		}
		#pragma omp parallel for
		for (int i = 0; i < size; i++) m[i] = v * vector[i];
		return lMatrix(m);
	}

	lVector& lVector::operator/=(double d) {
		if (!d) throw ERROR_IN_DIVISION_BY_ZERO;
		d = log2(d);
		normalized = false;
		if (deltaIndex) vector[deltaIndex - 1] -= d;
		else {
			#pragma omp parallel for
			for (int i = 0; i < size; i++) vector[i] -= d;
		}
		return *this;
	}

	lVector lVector::operator*(const lVector &v) const {
		if (size != v.size) throw ERROR_IN_PARAMETER_EXCEPTION;
		lVector p;
		p.size = size;
		p.normalized = false;
		p.vector.assign(size, -INFINITY);
		if (deltaIndex || v.deltaIndex) {
			p.deltaIndex = (deltaIndex) ? deltaIndex : v.deltaIndex;
			p.vector[p.deltaIndex - 1] = vector[p.deltaIndex - 1] + v.vector[p.deltaIndex - 1];
		}
		else {
			#pragma omp parallel for
			for (int i = 0; i < size; i++) p.vector[i] = vector[i] + v.vector[i];
		}
		return p;
	}

	lVector lVector::operator*(const lMatrix &m) const {
		if (size != m.getNRows()) throw ERROR_IN_PARAMETER_EXCEPTION;
		lVector p;
		p.size = m.getNColumns();
		p.normalized = false;
		p.vector.assign(p.size, 0.0);
		for (int i = 0; i < p.size; i++) {
			double sum = 0.0;
			#pragma omp parallel for reduction(+:sum)
			for (int j = 0; j < size; j++) {
				sum += exp2(vector[j] + m[j].vector[i]);
			}
			p.vector[i] = log2(sum);
		}
		return p;
	}

	std::ostream& operator<<(std::ostream &stream, const lVector &ob) {
		stream << "[ ";
		stream << std::fixed;
		for (double n : ob.vector) stream << exp2(n) << " ";
		stream << "]";
		return stream;
	}

	std::istream& operator>>(std::istream &stream, lVector &ob) {
		double d;
		std::vector<double> v;
		if (stream.get() == '[') {
			while (stream >> d) v.push_back(d);
			stream.clear();
			if (stream.get() == ']') ob.setVector(v);
			else stream.setstate(std::ios::failbit);
		}
		else stream.setstate(std::ios::failbit);
		return stream;
	}

	/* MATRIX */
	
	std::ostream& operator<<(std::ostream &stream, const Matrix<lVector> &ob) {
		size_t size = ob.matrix.size() - 1;
		stream << "[" << ob.matrix[0] << std::endl;
		for (size_t i = 1; i < size; i++) stream << " " << ob.matrix[i] << std::endl;
		stream << " " << ob.matrix[size] << "]";
		return stream;
	}

	std::ostream& operator<<(std::ostream &stream, const Matrix<std::vector<double>> &ob) {
		stream << "[[";
		stream << std::fixed;
		for (size_t j = 0; j < ob.matrix.size(); j++) stream << " " << ob.matrix[j][0];
		stream << " ]";
		for (size_t i = 1; i < ob.matrix[0].size(); i++) {
			stream << std::endl << " [";
			for (size_t j = 0; j < ob.matrix.size(); j++) stream << " " << ob.matrix[j][i];
			stream << " ]";
			
		}
		stream << "]";
		return stream;
	}

	std::istream& operator>>(std::istream &stream, Matrix<lVector> &ob) {
		std::string line;
		do {
			std::getline(stream, line);
			if (line.size() < 3) {
				stream.setstate(std::ios::failbit);
				break;
			}
			line.erase(0, 1);
			std::istringstream sstream(line);
			lVector p;
			sstream >> p;
			if (sstream.fail()) {
				stream.setstate(std::ios::failbit);
				break;
			}
			else ob.matrix.push_back(p);
		} while (line.compare(line.length() - 2, 2, "]]"));
		return stream;
	}

	/* LMATRIX */

	lMatrix::lMatrix(size_t r, size_t c, bool casual) {
		rows = r;
		cols = c;
		if (casual) {
			std::random_device rand_dev;
			std::mt19937 generator(rand_dev());
			std::uniform_real_distribution<double> distribution(0.001, 0.999);
			for (int i = 0; i < rows; i++) {
				std::vector<double> v = std::vector<double>(cols);
				for (int j = 0; j < cols; j++) v[j] = distribution(generator);
				matrix.push_back(lVector(v, true));
			}
		}
		else matrix = std::vector<lVector>(rows, lVector(cols));
	}

	lMatrix::lMatrix(const std::vector<lVector> &m, size_t index) {
		rows = m.size();
		cols = m[0].getSize();
		colIndex = index;
		matrix = m;
	}

	lMatrix::lMatrix(const sMatrix &m) {
		rows = m.getNRows();
		cols = m.getNColumns();
		colIndex = m.isColumn();
		matrix = std::vector<lVector>(rows);
		#pragma omp parallel for
		for (int i = 0; i < rows; i++) matrix[i] = m.getRow(i + 1); 
	}

	bool lMatrix::isSimilar(const lMatrix &m) const {
		if (rows != m.rows || cols != m.cols) return false;
		for (int i = 0; i < rows; i++) {
			if (!matrix[i].isSimilar(m.matrix[i])) return false;
		}
		return true;
	}

	sMatrix lMatrix::normalize() {
		double sum = 0.0;
		#pragma omp parallel for reduction(+:sum)
		for (int i = 0; i < rows; i++) sum += matrix[i].getSum();
		if (sum) {
			#pragma omp parallel for
			for (int i = 0; i < rows; i++) matrix[i] /= sum;
		}
		return sMatrix(*this);
	}

	lVector lMatrix::operator*(const lVector &v) const {
		std::vector<double> p(rows);
		#pragma omp parallel for
		for (int i = 0; i < rows; i++) p[i] = (matrix[i] * v).getSum();
		return lVector(p);
	}

	lMatrix lMatrix::operator*(const lMatrix &m) const {
		bool test = true;
		std::vector<lVector> temp(rows);
		temp[0] = matrix[0] * m[0];
		size_t delta = temp[0].getDelta();
		#pragma omp parallel for reduction(&&: test)
		for (int i = 1; i < rows; i++) {
			temp[i] = matrix[i] * m[i];
			test = temp[i].getDelta() == delta;
		}
		if (test) return lMatrix(temp, delta);
		return lMatrix(temp);
	}

	/* SMATRIX */

	sMatrix::sMatrix(size_t r, size_t c) {
		rows = r;
		cols = c;
		matrix = std::vector<std::vector<double>>(cols, std::vector<double>(rows, 0.0));
	}

	sMatrix::sMatrix(const lMatrix &m) {
		rows = m.getNRows();
		cols = m.getNColumns();
		colIndex = m.isColumn();
		matrix = std::vector<std::vector<double>>(cols, std::vector<double>(rows, 0.0));
		if (colIndex) {
			#pragma omp parallel for
			for (int i = 0; i < rows; i++) matrix[colIndex - 1][i] = m[i][colIndex - 1];
		}
		else {
			#pragma omp parallel for
			for (int i = 0; i < rows; i++) {
				for (int j = 0; j < cols; j++) {
					matrix[j][i] = m[i][j];
				}
			}
		}
	}

	lVector sMatrix::getRow(size_t index) const {
		std::vector<double> v(cols);
		#pragma omp parallel for
		for (int i = 0; i < cols; i++) v[i] = matrix[i][index - 1];
		return lVector(v, true);
	}

	sMatrix& sMatrix::operator+=(const sMatrix &m) {
		if (rows != m.rows || cols != m.cols) throw ERROR_IN_PARAMETER_EXCEPTION;
		if (m.colIndex) {
			#pragma omp parallel for
			for (int i = 0; i < rows; i++) matrix[m.colIndex - 1][i] += m.matrix[m.colIndex - 1][i];
		}
		else {
			for (int i = 0; i < cols; i++) {
				#pragma omp parallel for
				for (int j = 0; j < rows; j++) {
					matrix[i][j] += m.matrix[i][j];
				}
			}
		}
		return *this;
	}

	/* LINK */

	Link::Link(size_t d) {
		if (d < 2) throw ERROR_IN_PARAMETER_EXCEPTION;
		cardinality = d;
		forwardMessage = lVector(cardinality);
		backwardMessage = lVector(cardinality);
	}
	
	void Link::setForwardMessage(const lVector &f) {
		if (f.getSize() != cardinality) throw ERROR_IN_PARAMETER_EXCEPTION;
		forwardMessage = f;
		if(child != nullptr) child->update(false);
	}

	void Link::setBackwardMessage(const lVector &b) {
		if (b.getSize() != cardinality) throw ERROR_IN_PARAMETER_EXCEPTION;
		backwardMessage = b;
		if(parent != nullptr) parent->update(true);
	}

	void Link::reset() {
		forwardMessage = lVector(cardinality);
		backwardMessage = lVector(cardinality);
	}

	/* SISOBLOCK */

	SISOBlock::SISOBlock(Link &in, Link &out, bool casual) : inputLink(&in), outputLink(&out), casualityMatrix(casual) { 
		matrix = lMatrix(inputLink->getCardinality(), outputLink->getCardinality(), casualityMatrix);
		outputLink->setElement(*this, true);
		inputLink->setElement(*this, false);
	}

	void SISOBlock::setElement(Link &in, Link &out, bool casual) {
		inputLink = &in;
		outputLink = &out;
		casualityMatrix = casual;
		matrix = lMatrix(inputLink->getCardinality(), outputLink->getCardinality(), casualityMatrix);
		outputLink->setElement(*this, true);
		inputLink->setElement(*this, false);
	}

	void SISOBlock::setMatrix(const lMatrix &m) {
		if (m.getNColumns() != matrix.getNColumns() || m.getNRows() != matrix.getNRows()) throw ERROR_IN_PARAMETER_EXCEPTION; 
		matrix = m;
	}

	void SISOBlock::setState(LearningState state) {
		if (state == LEARNING) tempLearningMatrix = sMatrix(matrix.getNRows(), matrix.getNColumns());
		else if (state == BATCH_LEARNING) listInput.clear();
		actualState = state;
	}

	void SISOBlock::putsOut(size_t index) { 
		if (index >= listOutput.size()) throw ERROR_IN_PARAMETER_EXCEPTION;
		inputLink->setBackwardMessage(matrix * listOutput[index]); 
	}

	void SISOBlock::update(bool fromDown) {
		if (fromDown) {
			if (actualState == BATCH_LEARNING) listOutput.push_back(outputLink->getBackwardMessage());
			else inputLink->setBackwardMessage(matrix * outputLink->getBackwardMessage());
		}
		else {
			if (actualState == BATCH_LEARNING) listInput.push_back(inputLink->getForwardMessage());
			else {
				if (actualState == LEARNING) {
					tempLearningMatrix += (matrix * (inputLink->getForwardMessage()).rowColProd(outputLink->getBackwardMessage())).normalize();
				}
				else outputLink->setForwardMessage(inputLink->getForwardMessage() * matrix);
			}
		}
	}

	bool SISOBlock::learn(size_t loops) {
		if (actualState == LEARNING) matrix = lMatrix(tempLearningMatrix);
		else if (actualState == BATCH_LEARNING) {
			if (listInput.size() != listOutput.size()) throw ERROR_IN_PARAMETER_EXCEPTION;
			size_t count = 0;
			lMatrix H1, H2 = matrix;
			do {
				count++;
				H1 = H2;
				sMatrix temp = sMatrix(inputLink->getCardinality(), outputLink->getCardinality());
				for (int i = 0; i < listInput.size(); i++) {
					temp += (H2 * (listInput[i].rowColProd(listOutput[i]))).normalize();
				}
				H2 = lMatrix(temp);
			} while (count < loops && !H1.isSimilar(H2));
			matrix = H2;
			if (count == 1) return true;
		}
		return false;
	}
	
	void SISOBlock::reset() {
		listInput.clear();
		listOutput.clear();
		actualState = NOT_LEARNING;
		matrix = lMatrix(inputLink->getCardinality(), outputLink->getCardinality(), casualityMatrix);
	}

	/* DIVERTER */

	lVector Diverter::inMoltiplicator(size_t index) {
		lVector in = inputs[index]->getForwardMessage();
		if (inputs.size() == 1) return in;
		size_t size = inMatrices[index].size();
		std::vector<double> v = std::vector<double>(size);
		#pragma omp parallel for
		for (int i = 0; i < size; i++) v[i] = in[inMatrices[index][i]];
		return lVector(v);
	}
	void Diverter::outMoltiplicator(size_t index, const lVector &out) {
		if (inputs.size() == 1) inputs[index]->setBackwardMessage(out);
		std::vector<double> v = std::vector<double>();
		for (size_t i = 0; i < inMatrices[index].size(); i++) {
			if (v.size() > inMatrices[index][i]) v[inMatrices[index][i]] += out[i];
			else v.push_back(out[i]);
		}
		inputs[index]->setBackwardMessage(lVector(v));
	}

	void Diverter::initialize() {
		size_t all = 1, actual = 1, size = inputs.size();
		#pragma omp parallel for reduction(*:all)
		for (int i = 0; i < size; i++) {
			all *= inputs[i]->getCardinality();
			inputs[i]->setElement(*this, false);
		}
		inMatrices = std::vector<std::vector<size_t>>(size);
		for (size_t i = 0; i < size; i++) {
			int temp = inputs[i]->getCardinality();
			all /= temp;
			for (int j = 0; j < actual; j++) {
				for (int k = 0; k < temp; k++) {
					for (int l = 0; l < all; l++) {
						inMatrices[i].push_back(k);
					}
				}
			}
			actual *= temp;
		}
		#pragma omp parallel for
		for (int i = 0; i < outputs.size(); i++) outputs[i]->setElement(*this, true);
	}

	Diverter::Diverter(std::vector<Link> &in, std::vector<Link> &out) {
		size_t sIn = in.size(), sOut = out.size();
		inputs = std::vector<Link*>(sIn);
		outputs = std::vector<Link*>(sOut);
		#pragma omp parallel for
		for (int i = 0; i < sIn; i++) inputs[i] = &in[i];
		#pragma omp parallel for
		for (int i = 0; i < sOut; i++) outputs[i] = &out[i];
		initialize();
	}

	void Diverter::execute() {
		size_t sizeI = inputs.size(), sizeO = outputs.size() - 1, size = sizeI + sizeO;
		std::vector<lVector> temp = std::vector<lVector>(size);
		temp[0] = inMoltiplicator(0);
		for (size_t i = 1; i < sizeI; i++) temp[i] = temp[i - 1] * inMoltiplicator(i);
		for (size_t i = 0; i < sizeO; i++) temp[sizeI + i] = temp[sizeI + i - 1] * outputs[i]->getBackwardMessage();
		outputs[sizeO]->setForwardMessage(temp[size - 1]);
		temp[size - 1] = outputs[sizeO]->getBackwardMessage();
		for (size_t i = 1; i < sizeO + 1; i++) {
			outputs[sizeO - i]->setForwardMessage(temp[size - i] * temp[size - 1 - i]);
			temp[size - 1 - i] = temp[size - i] * outputs[sizeO - i]->getBackwardMessage();
		}
		for (size_t i = 1; i < sizeI; i++) {
			outMoltiplicator(sizeI - i, temp[sizeI - i] * temp[sizeI - 1 - i]);
			temp[sizeI - 1 - i] = temp[sizeI - i] * inMoltiplicator(sizeI - i);
		}
		outMoltiplicator(0, temp[0]);
	}

	/* SOURCE */

	Source::Source(Link &out, const lVector *p) : outputLink(&out) {
		size_t size = outputLink->getCardinality();
		outputLink->setElement(*this, true);
		if (p == nullptr || p->getSize() != size) prior = lVector(size);
		else {
			prior = *p;
			outputLink->setForwardMessage(prior);
		}
	}

	void Source::setState(LearningState state) {
		if(state != NOT_LEARNING) tempLearningPrior.assign(outputLink->getCardinality(), 0.0);
		actualState = state;
	}

	void Source::update(bool fromDown) {
		if (actualState != NOT_LEARNING) {
			std::vector<double> v = outputLink->getBackwardMessage().getVector();
			#pragma omp parallel for
			for (int i = 0; i < v.size(); i++) tempLearningPrior[i] += v[i];
		}
	}

	void Source::learn() {
		if (actualState != NOT_LEARNING) {
			prior = lVector(tempLearningPrior, true);
			outputLink->setForwardMessage(prior);
		}
	}

	void Source::reset() {
		tempLearningPrior.clear();
		prior = lVector(prior.getSize());
	}
	
} // namespace FGrn