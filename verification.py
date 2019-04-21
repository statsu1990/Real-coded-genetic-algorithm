import RealCodedGeneticAlgorithm as rcga

import numpy as np
import matplotlib.pyplot as plt

class Minimize_L2:
    def __init__(self, gene_num, 
                 initial_min, initial_max, 
                 population, 
                 crossover_num, child_num,
                 initial_expantion_rate=1, learning_rate=0.1, 
                 seed=None):

        # rcga class
        flag = 1

        if flag == 0:
            self.ga = rcga.RealCodecGA_JGG_SPX(gene_num=gene_num, 
                                                    initial_min=initial_min, initial_max=initial_max, 
                                                    population=population, 
                                                    crossover_num=crossover_num, child_num=child_num,
                                                    evaluation_func=self.evaluation_func, 
                                                    seed=seed)
        elif flag == 1:
            self.ga = rcga.RealCodecGA_JGG_AREX(gene_num=gene_num, 
                                                evaluation_func=self.evaluation_func, 
                                                initial_min=initial_min, initial_max=initial_max, 
                                                population=population, 
                                                crossover_num=crossover_num, child_num=child_num, 
                                                initial_expantion_rate=initial_expantion_rate, learning_rate=learning_rate, 
                                                seed=seed)

        return

    def evaluation_func(self, genes):
        ev = np.average(np.square(genes), axis=1)
        return ev

    def run(self, step_num, print_evaluation=True, print_fig=True):
        if print_evaluation:
            print('generation, best_evaluation, diversity')
        #
        best_evals = []
        for i in range(step_num):
            self.ga.generation_step()
            best_evals.append(self.ga.best_evaluation)
            diversity = self.ga.calc_diversity()

            if print_evaluation:
                print('{0}, {1}, {2}'.format(i+1, self.ga.best_evaluation, diversity))

        #
        self.__summary(best_evals, self.ga)

        if print_fig:
            self.__conv_fig(best_evals, skip_display=2)

        return best_evals, self.ga.best_gene, self.ga.best_evaluation

    @staticmethod
    def __summary(best_evals, ga):
        # best
        print('<summary>')
        print(' best_evals, {0}'.format(best_evals[-1]))
        print(' best_gene, {0}'.format(ga.best_gene))
        return

    @staticmethod
    def __conv_fig(best_evals, skip_display=2):
        # fig
        display_start_idx = skip_display if len(best_evals) > skip_display else 0
        #display_start_idx = int((len(best_evals)+1)*(1-display_rate))
        #
        x = np.arange(display_start_idx+1, len(best_evals)+1)
        plt.plot(x, np.array(best_evals)[display_start_idx:], marker="o", markersize=2)
        plt.xlabel('generation')
        plt.ylabel('evaluation')
        plt.yscale('log')
        plt.show()
        return

def test_Minimize_L2_1():
    '''
    gene_num = 10
    population = np.maximum(10, int(np.sqrt(gene_num) * 30)) # 密度
    crossover_num = np.maximum(3, int(population / 50)) # 入れ替わり率
    child_num = int(crossover_num * 1.5)
    step_num = int(population / crossover_num) * 50 # 総変わり数
    '''
    '''
    gene_num = 10
    population = np.maximum(10, int(np.sqrt(gene_num) * 1000)) # 密度
    crossover_num = np.maximum(3, int(population / 1000)) # 入れ替わり率
    child_num = int(crossover_num * 10)
    step_num = int(population / crossover_num * 10) # 総変わり数
    '''
    '''
    gene_num = 50
    population = np.maximum(10, int(np.sqrt(gene_num) * 1000)) # 密度
    crossover_num = np.maximum(3, int(population / 1000)) # 入れ替わり率
    child_num = int(crossover_num * 100)
    step_num = int(population / crossover_num * 10) # 総変わり数
    '''
    '''
    gene_num = 100
    population = np.maximum(10, int(gene_num * 10)) # 密度
    crossover_num = np.maximum(3, int(np.sqrt(population) / 10)) # 入れ替わり率
    child_num = int(crossover_num * 30)
    step_num = int(population / crossover_num * 10) # 総変わり数
    '''
    '''
    gene_num = 100
    population = np.maximum(10, int(np.sqrt(gene_num) * 100)) # 密度
    crossover_num = np.maximum(3, int(population / 100)) # 入れ替わり率
    child_num = int(crossover_num * 20)
    step_num = int(population / crossover_num * 20) # 総変わり数
    '''
    gene_num = 100
    population = gene_num * 3 # 密度
    crossover_num = None # 入れ替わり率
    child_num = None
    step_num = 500 # 総変わり数

    '''
    gene_num = 200
    population = np.maximum(10, int(gene_num * 10)) # 密度
    crossover_num = gene_num + 1
    child_num = int(crossover_num * 3.5)
    step_num = int(population / crossover_num * 100) # 総変わり数
    '''

    vr = Minimize_L2(gene_num=gene_num, 
                    initial_min=-1, initial_max=1, 
                    population=population, 
                    crossover_num=crossover_num, child_num=child_num, 
                    seed=1000)
    _, _, best_eval = vr.run(step_num=step_num, print_evaluation=True, print_fig=True)

    #
    print()
    print('gene_num:{0}'.format(gene_num))
    print('population:{0}'.format(population))
    print('crossover_num:{0}'.format(crossover_num))
    print('child_num:{0}'.format(child_num))
    print('step_num:{0}'.format(step_num))

    return


from sklearn.linear_model import LinearRegression, Ridge

class LinearLeastSquares_withRCGA:
    def __init__(self, x_num, sample_num, coefs=None, bias=None, error_scale=0.1, seed=None):
        self.x_num = x_num
        self.sample_num = sample_num
        
        # random
        self.seed = seed
        np.random.seed(seed)

        # coef, bias
        self.coefs = coefs if coefs is not None else np.random.rand(x_num) * 2.0 - 1.0
        self.bias = bias if bias is not None else np.random.rand() * 2.0 - 1.0

        # samples
        self.x = np.random.rand(sample_num, x_num)
        #
        self.y = np.dot(self.x, self.coefs)  + self.bias
        error = np.random.normal(loc=0.0, scale=error_scale, size=sample_num)
        self.y = self.y + error

        #
        self.ga = None

        return

    def evaluation_func_forGA(self, genes):
        '''
        rmse of y
        coef = bias_coefs[1:]
        bias = bias_coefs[0]
        '''
        vec_f = np.vectorize(self.__eval_func, signature='(m)->()')
        evals = vec_f(genes)

        return evals

    def __eval_func(self, bias_coefs):
        '''
        rmse of y
        coef = bias_coefs[1:]
        bias = bias_coefs[0]
        '''
        pre_y = np.dot(self.x, bias_coefs[1:]) + bias_coefs[0]
        eval = np.average(np.square(self.y - pre_y))
        return eval

    def make_ga(self, initial_min=-1, initial_max=1, 
                population=20, 
                crossover_num=None, child_num=None, 
                initial_expantion_rate=None, learning_rate=None, 
                seed=None):
        self.ga = rcga.RealCodecGA_JGG_AREX(gene_num=self.x_num+1, 
                                            evaluation_func=self.evaluation_func_forGA, 
                                            initial_min=initial_min, initial_max=initial_max, 
                                            population=population, 
                                            crossover_num=crossover_num, child_num=child_num, 
                                            initial_expantion_rate=initial_expantion_rate, learning_rate=learning_rate, 
                                            seed=seed)
        return

    def run(self, step_num, print_evaluation=True, print_fig=True):
        if print_evaluation:
            print('generation, best_evaluation, diversity')
        #
        best_evals = []
        for i in range(step_num):
            self.ga.generation_step()
            best_evals.append(self.ga.best_evaluation)
            diversity = self.ga.calc_diversity()

            if print_evaluation:
                print('{0}, {1}, {2}'.format(i+1, self.ga.best_evaluation, diversity))

        # summary
        self.__summary(best_evals, self.ga)
        # figure
        if print_fig:
            self.__conv_fig(best_evals, skip_display=2)

        # answer
        self.__answer()

        return best_evals, self.ga.best_gene, self.ga.best_evaluation

    def __answer(self):
        print('<Direct method>')
        lr = LinearRegression()
        lr.fit(self.x, self.y)
        bs = np.array([lr.intercept_] + lr.coef_.tolist())
        print(' eval, {0}'.format(self.__eval_func(bs)))
        print(' gene, {0}'.format(bs))
        print()

        print('<Gradient method>')
        lr = Ridge(alpha=0.0, max_iter=10000, tol=0.0001)
        lr.fit(self.x, self.y)
        bs = np.array([lr.intercept_] + lr.coef_.tolist())
        print(' eval, {0}'.format(self.__eval_func(bs)))
        print(' gene, {0}'.format(bs))
        print()

        print('<answer>')
        bs = np.array([self.bias]+self.coefs.tolist())
        print(' ref eval, {0}'.format(self.__eval_func(bs)))
        print(' gene, {0}'.format(bs))


        return

    @staticmethod
    def __summary(best_evals, ga):
        # best
        print('<Genetic Algorithm>')
        print(' best_evals, {0}'.format(best_evals[-1]))
        print(' best_gene, {0}'.format(ga.best_gene))
        return

    @staticmethod
    def __conv_fig(best_evals, skip_display=2):
        # fig
        display_start_idx = skip_display if len(best_evals) > skip_display else 0
        #display_start_idx = int((len(best_evals)+1)*(1-display_rate))
        #
        x = np.arange(display_start_idx+1, len(best_evals)+1)
        plt.plot(x, np.array(best_evals)[display_start_idx:], marker="o", markersize=2)
        plt.xlabel('generation')
        plt.ylabel('evaluation')
        plt.yscale('log')
        plt.show()
        return

def test_LinearLeastSquares_withRCGA():
    x_num = 100
    sample_num = x_num * 50
    coefs=None
    bias=None
    error_scale=0.01
    seed=10
    #
    initial_min=-1
    initial_max=1
    population=x_num*10
    crossover_num=None
    child_num=None
    initial_expantion_rate=None
    learning_rate=None
    ga_seed=11
    #
    step_num=1000
    
    '''
    x_num = 200
    sample_num = x_num * 50
    coefs=None
    bias=None
    error_scale=0.1
    seed=10
    #
    initial_min=-1
    initial_max=1
    crossover_num=int(np.sqrt(x_num))*3
    population=crossover_num*30
    child_num=4*crossover_num
    initial_expantion_rate=1.1
    learning_rate=0.1
    ga_seed=11
    #
    step_num=10000
    '''

    #
    vr = LinearLeastSquares_withRCGA(x_num=x_num, sample_num=sample_num, 
                                     coefs=coefs, bias=bias, 
                                     error_scale=error_scale, 
                                     seed=seed)
    #
    vr.make_ga(initial_min=initial_min, initial_max=initial_max, 
                population=population, 
                crossover_num=crossover_num, child_num=child_num, 
                initial_expantion_rate=initial_expantion_rate, learning_rate=learning_rate, 
                seed=ga_seed)

    _, _, best_eval = vr.run(step_num=step_num, print_evaluation=True, print_fig=True)

    return