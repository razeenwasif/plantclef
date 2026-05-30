import Topbar from "@/components/Topbar";
import Hero from "@/components/Hero";
import Pipeline from "@/components/Pipeline";
import Stats from "@/components/Stats";
import Abstract from "@/components/Abstract";
import Tasks from "@/components/Tasks";
import Dataset from "@/components/Dataset";
import ChartsSection from "@/components/ChartsSection";
import Leaderboard from "@/components/Leaderboard";
import Baselines from "@/components/Baselines";
import TeamBlock from "@/components/TeamBlock";
import BibTeX from "@/components/BibTeX";
import Footer from "@/components/Footer";

export default function Page() {
  return (
    <>
      <Topbar />
      <Hero />
      <Pipeline />
      <Stats />
      <Abstract />
      <Tasks />
      <Dataset />
      <ChartsSection />
      <Leaderboard />
      <Baselines />
      <TeamBlock />
      <BibTeX />
      <Footer />
    </>
  );
}
