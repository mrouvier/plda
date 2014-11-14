#include <iostream>
#include <map>
#include <fstream>
#include <vector>
#include <Eigen/Dense>
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/unordered_map.hpp>
#include <boost/program_options.hpp>

using namespace std;
using namespace boost;
using namespace boost::program_options;



static inline std::string &ltrim(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), std::not1(std::ptr_fun<int, int>(std::isspace))));
    return s;
}


static inline std::string &rtrim(std::string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(), std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
    return s;
}


static inline std::string &trim(std::string &s) {
    return ltrim(rtrim(s));
}


boost::unordered_map<string, Eigen::VectorXd> read_x_file(const string& __file) {
    boost::unordered_map<string, Eigen::VectorXd> h;

    string line;
    fstream f(__file.c_str());
    while(getline(f, line)) {
        line = trim(line);
        vector<string> vec;
        split(vec, line, is_any_of(" "));

        Eigen::VectorXd v(vec.size()-1);
        for(int i = 1; i < vec.size(); i++) {
            //cerr<<"DEBUG : "<<lexical_cast<float>( vec[i] )<<endl;
            v[i-1] = lexical_cast<double>( vec[i] );
        }
        h[ vec[0] ] = v;
    }
    f.close();


    return h;
}

Eigen::MatrixXd covariance(vector< vector<string> > ar_c, boost::unordered_map<string, Eigen::VectorXd> h_x) {
    int size = h_x[ ar_c[0][0] ].size();
    Eigen::MatrixXd cov(size, size); cov.setZero();
    Eigen::VectorXd mean = Eigen::VectorXd::Zero(size);

    for(boost::unordered_map<string, Eigen::VectorXd>::iterator a = h_x.begin(); a != h_x.end(); a++) {
        mean += a->second;
    }
    mean /= (double)h_x.size();

    for(boost::unordered_map<string, Eigen::VectorXd>::iterator a = h_x.begin(); a != h_x.end(); a++) {
        Eigen::VectorXd m = a->second - mean;
        cov += ( m * m.transpose() );
    }

    cov /= (double) h_x.size();

    return cov;
}



double scoring_plda(Eigen::VectorXd x, Eigen::VectorXd y, Eigen::MatrixXd p, Eigen::MatrixXd q) {
    double a = ((x.transpose() * q * x) + (y.transpose() * q * y) + (2*x.transpose() * p * y))[0];
    return a;
}

Eigen::MatrixXd intra_covariance(vector< vector<string> > ar_c, boost::unordered_map<string, Eigen::VectorXd> h_x) {
    int size = h_x[ ar_c[0][0] ].size();

    int counter = 0;

    Eigen::MatrixXd cov(size, size); cov.setZero();

    for(int i = 0; i < ar_c.size(); i++) {
        Eigen::VectorXd mean = Eigen::VectorXd::Zero(size);

        for(int j = 0; j < ar_c[i].size(); j++) {
            mean += h_x[ar_c[i][j]];
        }
        mean /= (double)ar_c[i].size();

        for(int j = 0; j < ar_c[i].size(); j++) {
            Eigen::VectorXd m = h_x[ar_c[i][j]] - mean;
            Eigen::MatrixXd a = (m * m.transpose());
            cov += a;
        }

        counter += ar_c[i].size();
    }

    cov /= (double) counter;
    return cov;
}

double lengthNorm(Eigen::VectorXd l) {

    double c = 0;

    for(int i = 0; i < l.size(); i++) {
        c += l[i]*l[i];    
    }

    return sqrt(c);
}

boost::unordered_map<string, Eigen::VectorXd> normalize_data_train(vector< vector<string> > ar_c, boost::unordered_map<string, Eigen::VectorXd> h_x, Eigen::VectorXd& mean, Eigen::MatrixXd& total_cov) {


    int size = h_x[ ar_c[0][0] ].size();
    mean = Eigen::VectorXd::Zero(size);
    Eigen::MatrixXd cov = intra_covariance(ar_c, h_x);

    //Eigenvalue - Eigenvectors
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(cov);
    Eigen::MatrixXd d = es.eigenvalues().asDiagonal();
    for(int i = 0; i < size; i++) d(i, i) = 1 / sqrt( d(i, i) );
    total_cov = d * es.eigenvectors().transpose();

    //Mean
    for(boost::unordered_map<string, Eigen::VectorXd>::iterator a = h_x.begin(); a != h_x.end(); a++) {
        mean += a->second;         
    }
    mean /= (double) h_x.size();


    //Normalize
    boost::unordered_map<string, Eigen::VectorXd> h;

    for(boost::unordered_map<string, Eigen::VectorXd>::iterator a = h_x.begin(); a != h_x.end(); a++) {
        Eigen::VectorXd m = total_cov * (a->second - mean); 
        h[ a->first ] = m / lengthNorm(m);
    }


    return h;
}

boost::unordered_map<string, Eigen::VectorXd> normalize_data_test(boost::unordered_map<string, Eigen::VectorXd> h_x, Eigen::VectorXd mean, Eigen::MatrixXd cov) {
    boost::unordered_map<string, Eigen::VectorXd> h;
    for(boost::unordered_map<string, Eigen::VectorXd>::iterator a = h_x.begin(); a != h_x.end(); a++) {
        Eigen::VectorXd m = cov * ( a->second - mean);
        h[ a->first ] = m / lengthNorm(m);
    }
    return h;
}

vector< vector<string> > read_file_covariance(const string& __file) {
    vector< vector<string> > ar;
    string line;

    fstream f(__file.c_str());
    while(getline(f, line)) {
        line = trim(line);
        vector<string> vec;
        split(vec, line, is_any_of(" "));
        ar.push_back( vec );
    }
    f.close();

    return ar;
}


void plda(vector< vector<string> > ar_c, boost::unordered_map<string, Eigen::VectorXd> h_x, Eigen::MatrixXd& V, Eigen::MatrixXd& Sigma, Eigen::VectorXd& mu, int rdim, int nb_iter) {


    int size = h_x[ ar_c[0][0] ].size();

    mu.resize(size);
    mu = Eigen::VectorXd::Zero(size);
    Eigen::MatrixXd S;
    //Eigen::MatrixXd Sigma;
    vector< Eigen::VectorXd > f;
    double N = (double)h_x.size();
    V = Eigen::MatrixXd::Random(size, rdim);


    //Mean
    for(boost::unordered_map<string, Eigen::VectorXd>::iterator a = h_x.begin(); a != h_x.end(); a++) {
        mu += a->second;
    }
    mu /= N;


    //First statistics
    for(int i = 0; i < ar_c.size(); i++) {
        Eigen::VectorXd temp = Eigen::VectorXd::Zero(size);

        for(int j = 0; j < ar_c[i].size(); j++) {
            Eigen::VectorXd x = h_x[ ar_c[i][j] ] ;
            x -= mu;
            temp += x;
        }
        //temp /= ar_c[i].size();
        f.push_back( temp );
    } 


    //Second statistics
    Eigen::MatrixXd data(h_x.size(), size);
    int counter = 0;
    for(boost::unordered_map<string, Eigen::VectorXd>::iterator a = h_x.begin(); a != h_x.end(); a++) {
        for(int i = 0; i < size; i++) {
            Eigen::VectorXd x = a->second;
            x -= mu;
            data(counter, i) = x(i);
        }
        counter += 1;
    }
    S = data.transpose() * data;

    Sigma = S/N;


    int dim_d = size;
    int dim_V = rdim;


    for(int iter = 0; iter < nb_iter; iter++) {

        //cerr<<"Iteration : "<<iter<<endl;

        Eigen::MatrixXd Eyy(dim_V, dim_V); Eyy.setZero();
        Eigen::MatrixXd T(dim_d, dim_V); T.setZero();


        Eigen::MatrixXd SV = Sigma.inverse() * V;
        Eigen::MatrixXd VSV = V.transpose() * SV;
        Eigen::MatrixXd Y_md(dim_V, dim_V); Y_md.setZero();

        for(int i = 0; i < ar_c.size(); i++) {
            float n = (float)ar_c[i].size();
            Eigen::MatrixXd M_i = (n*VSV + Eigen::MatrixXd::Identity(dim_V, dim_V) ).inverse();
            Eigen::VectorXd Ey_i = M_i * ( SV.transpose() * f[i] );
 
            Eyy += n*( M_i + (Ey_i * Ey_i.transpose()) );

            T += f[i] * Ey_i.transpose();
            Y_md += (M_i + Ey_i*Ey_i.transpose());
        }

        Y_md = Y_md / (double)ar_c.size();

        V = T * Eyy.inverse();

        Sigma =  (S - (V * T.transpose()))/N;
        V = V * Y_md.llt().matrixL();

        /*
        double sum = 0;
        for(int i = 0; i < 40; i++) {
            for(int j = 0; j < ar_c[i].size(); j++) {
                Eigen::MatrixXd s = Sigma;
                Eigen::VectorXd x = h_x[ ar_c[i][j] ];
                Eigen::VectorXd z = V.transpose() * (x-mu);

                double a = (( x - mu - V*z).transpose() * s.inverse() * ( x - mu - V*z ))[0];

                sum += a;
            }
        }
        cerr<<"Total : "<<sum<<endl;
        */

    }



}



void lda_train(vector< vector<string> > ar_c, boost::unordered_map<string, Eigen::VectorXd>  h_x, Eigen::VectorXd& mean, Eigen::MatrixXd& lda, int dim) {

    int size = h_x[ ar_c[0][0] ].size();

    mean.resize(size);
    mean = Eigen::VectorXd::Zero(size);

    boost::unordered_map<string, Eigen::VectorXd> h;

    //Mean
    for(boost::unordered_map<string, Eigen::VectorXd>::iterator a = h_x.begin(); a != h_x.end(); a++) {
        mean +=  a->second;
    }
    mean /= h_x.size();

    //Substract mean
    for(boost::unordered_map<string, Eigen::VectorXd>::iterator a = h_x.begin(); a != h_x.end(); a++) {
        Eigen::VectorXd t = a->second;
        h[ a->first ] = t - mean;
    }


    Eigen::MatrixXd intra_cov = intra_covariance(ar_c, h);
    Eigen::MatrixXd cov = covariance(ar_c, h);
    Eigen::MatrixXd inter_cov = cov - intra_cov;

    //w = cov^{-1} * between
    Eigen::MatrixXd t = intra_cov.llt().matrixL();
    t = t.inverse();
    Eigen::MatrixXd w = t * inter_cov * t.transpose();

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(w);

    Eigen::MatrixXd eigenv = es.eigenvectors();

    lda.resize(eigenv.rows(), dim);
    lda.setZero();

    for(int i = 0; i < eigenv.rows(); i++) {
        int counter = 0;
        for(int j = eigenv.cols()-dim; j < eigenv.cols(); j++) {
            lda(i, counter) = eigenv(i, j);
            counter += 1;
        }
    }

    lda = lda.transpose() * t;


}

boost::unordered_map<string, Eigen::VectorXd> lda_test(boost::unordered_map<string, Eigen::VectorXd> h_x, Eigen::VectorXd mean, Eigen::MatrixXd lda) {

    boost::unordered_map<string, Eigen::VectorXd> h;

    for(boost::unordered_map<string, Eigen::VectorXd>::iterator a = h_x.begin(); a != h_x.end(); a++) {
        Eigen::VectorXd t = a->second;
        t -= mean;
        Eigen::VectorXd x = lda * t;
        h[ a->first ] = x;
    }

    return h;

}


int main(int argc, char** argv) {

    int lda_dim, plda_dim, plda_nb_iter, lw_nb_iter;
    string train, label, test;

    options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("lda_dim", value<int>(&lda_dim)->default_value(0), "lda reduction dimension -- if 0 there is no lda reduction")
        ("plda_dim", value<int>(&plda_dim)->default_value(90), "plda reduction dimension")
        ("plda_nb_iter", value<int>(&plda_nb_iter)->default_value(10), "plda number iteration")
        ("lw_nb_iter", value<int>(&lw_nb_iter)->default_value(2), "Length Withened (LW) number iteration -- if 0 there is no lw normalization")
        ("train", value<string>(&train)->required(), "training corpus")
        ("label", value<string>(&label)->required(), "label corpus")
        ("test", value<string>(&test)->required(), "testing corpus")
        ;

    positional_options_description p;
    p.add("train", 1);
    p.add("label", 1);
    p.add("test", 1);

    variables_map vm;

    try {
        store(command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
        notify(vm);    
    }
    catch(...) {
        cout << desc << endl;
        return 0;
    }

    if (vm.count("help")) {
        cout << desc << endl;
        return 0;
    }


    boost::unordered_map<string, Eigen::VectorXd> h_x = read_x_file(train);
    boost::unordered_map<string, Eigen::VectorXd> h = read_x_file(test);
    vector< vector<string> > ar_c_train = read_file_covariance(label);

    int size = h_x[ ar_c_train[0][0] ].size();

    if(lda_dim != 0) {
        if(lda_dim <= size) {

            Eigen::VectorXd mean_lda;
            Eigen::MatrixXd lda;
            lda_train(ar_c_train, h_x, mean_lda, lda, lda_dim);
            h_x = lda_test(h_x, mean_lda, lda);
            h = lda_test(h, mean_lda, lda);

        }
    }



    for(int i = 0; i < lw_nb_iter; i++) {
        Eigen::VectorXd mean;
        Eigen::MatrixXd total_cov;

        h_x = normalize_data_train(ar_c_train, h_x, mean, total_cov);
        h = normalize_data_test(h, mean, total_cov);
    }


    Eigen::VectorXd mu(1);
    Eigen::MatrixXd V;
    Eigen::MatrixXd Sigma;
    plda(ar_c_train, h_x, V, Sigma, mu, plda_dim, plda_nb_iter);

    Eigen::MatrixXd Sigma_wc = Sigma;
    Eigen::MatrixXd Sigma_ac = V * V.transpose();
    Eigen::MatrixXd Sigma_tot = Sigma_wc + Sigma_ac;

    Eigen::MatrixXd Lambda = -1*(Sigma_wc+2*Sigma_ac).inverse() + Sigma_wc.inverse();
    Eigen::MatrixXd Gamma  = -1*(Sigma_wc+2*Sigma_ac).inverse() - Sigma_wc.inverse() + 2*Sigma_tot.inverse();


    for(boost::unordered_map<string, Eigen::VectorXd>::iterator a = h.begin(); a != h.end(); a++) {
        for(boost::unordered_map<string, Eigen::VectorXd>::iterator b = h.begin(); b != h.end(); b++) {
            cout<< a->first << " " << b->first << " " << scoring_plda( a->second-mu, b->second-mu, Gamma, Lambda)<<endl;
        }
     }

    return 0;
}

