import { Link } from 'react-router-dom';
import { ArrowRight } from 'lucide-react';
import { Button } from '@components/ui';

export const Home = () => {
  const steps = [
    {
      number: 1,
      title: 'Upload PDF',
      description: 'Upload your PDF document (max 10MB)',
    },
    {
      number: 2,
      title: 'Select Voices',
      description: 'Choose AI voices for host and guest',
    },
    {
      number: 3,
      title: 'Review Script',
      description: 'Edit the generated conversation if needed',
    },
    {
      number: 4,
      title: 'Download',
      description: 'Get your podcast audio file',
    },
  ];

  return (
    <div className="space-y-16">
      {/* Hero Section */}
      <section className="text-center space-y-6 py-12">
        <h1 className="text-5xl md:text-6xl font-bold text-gray-900 leading-tight">
          Transform PDFs into
          <br />
          <span className="bg-gradient-to-r from-primary-600 to-secondary-600 bg-clip-text text-transparent">
            Engaging Podcasts
          </span>
        </h1>

        <p className="text-xl text-gray-600 max-w-2xl mx-auto">
          Turn any PDF document into a natural, AI-powered podcast conversation
          in minutes. Perfect for learning, content creation, and accessibility.
        </p>

        <div className="flex flex-col sm:flex-row gap-4 justify-center pt-4">
          <Link to="/generate">
            <Button size="xl" icon={ArrowRight} iconPosition="right">
              Get Started
            </Button>
          </Link>
          <Link to="/projects">
            <Button size="xl" variant="outline">
              View Projects
            </Button>
          </Link>
        </div>
      </section>

      {/* How It Works */}
      <section>
        <h2 className="text-3xl font-bold text-center text-gray-900 mb-12">
          How It Works
        </h2>
        <div className="grid md:grid-cols-4 gap-8">
          {steps.map((step, index) => (
            <div key={index} className="relative">
              <div className="flex flex-col items-center text-center">
                <div className="w-16 h-16 bg-gradient-to-br from-primary-600 to-secondary-600 rounded-full flex items-center justify-center text-white text-2xl font-bold mb-4 shadow-lg">
                  {step.number}
                </div>
                <h3 className="text-xl font-semibold text-gray-900 mb-2">
                  {step.title}
                </h3>
                <p className="text-gray-600">{step.description}</p>
              </div>
              {index < steps.length - 1 && (
                <div className="hidden md:block absolute top-8 left-1/2 w-full h-0.5 bg-gradient-to-r from-primary-600 to-secondary-600 opacity-30" />
              )}
            </div>
          ))}
        </div>
      </section>
    </div>
  );
};

export default Home;
